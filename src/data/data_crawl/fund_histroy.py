import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import json
import pandas as pd
from time import sleep
import logging
from pymysql import connect, Error

logging.basicConfig(level=logging.INFO)


class FundHistoryCrawl(object):
    def __init__(self, max_requests_per_second=5):
        self.conn = connect(host='localhost', port=3306, database='financedb', user='root', password='mysql')
        self.cursor = self.conn.cursor()
        self.max_requests_per_second = max_requests_per_second
        self.semaphore = threading.Semaphore(max_requests_per_second)

    def _reset_auto_increment(self):
        sql_str = "ALTER TABLE fund_history AUTO_INCREMENT = 1;"
        self.cursor.execute(sql_str)

    def init_database(self, fund_code):
        self._delete_existing_data(fund_code)
        self._reset_auto_increment()
        self.conn.commit()

    def _delete_existing_data(self, fund_code):
        self.cursor.execute("DELETE FROM fund_history WHERE fund_code = %s;", (fund_code,))

    def do_crawl(self, fund_codes):
        """
        查询全部历史净值
        :return: 查询结果字典，成功或者失败
        """
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(self._crawl_single_fund, fund_code) for fund_code in fund_codes]
            results = []
            for future in as_completed(futures):
                results.append(future.result())
            return results

    def _crawl_single_fund(self, fund_code, page_range: range = None) -> dict:
        try:
            total_count = self._fetch_total_count(fund_code)
            page_total = (total_count + 19) // 20  # 等价于 math.ceil(total_count / 20)
            logging.info(f"基金代码 {fund_code} 的总页数为 {page_total}")

            if page_range is None:
                page_range = range(1, page_total + 1)

            df_tmp = pd.DataFrame()
            for single_page in page_range:
                logging.info(f"基金代码 {fund_code} 正在处理第 {single_page} 页数据")
                page_data = self._fetch_page_data(single_page, fund_code)
                df_tmp = pd.concat([df_tmp, page_data], ignore_index=True)
                sleep(1 / self.max_requests_per_second)  # 控制请求频率

            df_tmp = self._merge_info(df_tmp, fund_code)
            df_tmp = self._clean_data(df_tmp)  # 清洗数据
            self._insert_data(df_tmp)

            return {"fund_code": fund_code, "message": "ok", "status": 200}
        except Exception as e:
            logging.error(f"Error with fund_code {fund_code}: {e}")
            return {"fund_code": fund_code, "message": "error", "status": 400}

    def _merge_info(self, df_data: pd.DataFrame, fund_code: str) -> pd.DataFrame:
        df_data["fund_code"] = fund_code
        df = df_data[["fund_code", "FSRQ", "DWJZ", "LJJZ", "JZZZL"]]
        return df

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # 确保数值列没有空字符串或非数值字符
        df['DWJZ'] = pd.to_numeric(df['DWJZ'], errors='coerce')
        df['LJJZ'] = pd.to_numeric(df['LJJZ'], errors='coerce')
        df['JZZZL'] = pd.to_numeric(df['JZZZL'], errors='coerce')
        df = df.fillna(0)  # 用 0 填充 NaN
        return df

    def _fetch_total_count(self, fund_code):
        with self.semaphore:
            url = f"https://api.fund.eastmoney.com/f10/lsjz"
            params = {
                "callback": "jQuery18304038998523093684_1586160530315",
                "fundCode": fund_code,
                "pageIndex": 1,
                "pageSize": 20
            }
            header = {"Referer": f"https://fundf10.eastmoney.com/jjjz_{fund_code}.html"}
            response = requests.get(url, headers=header, params=params)
            response.raise_for_status()

            json_data = json.loads(response.text[41:-1])
            logging.info(f"基金代码 {fund_code} 初次执行结果：\n{json_data}")
            return json_data.get("TotalCount")

    def _fetch_page_data(self, page, fund_code) -> pd.DataFrame:
        with self.semaphore:
            url = f"https://api.fund.eastmoney.com/f10/lsjz"
            params = {
                "callback": "jQuery18304038998523093684_1586160530315",
                "fundCode": fund_code,
                "pageIndex": page,
                "pageSize": 20
            }
            header = {"Referer": f"https://fundf10.eastmoney.com/jjjz_{fund_code}.html"}
            response = requests.get(url, headers=header, params=params)
            response.raise_for_status()

            json_data = json.loads(response.text[41:-1])
            list_date_data = json_data.get("Data", {"LSJZList": None}).get("LSJZList")
            df = pd.DataFrame(list_date_data)

            # 确认 DataFrame 列名是否与数据库列名匹配，并进行修正
            expected_columns = ['FSRQ', 'DWJZ', 'LJJZ', 'JZZZL']
            for col in expected_columns:
                if col not in df.columns:
                    df[col] = None  # 添加缺失的列并填充默认值

            return df

    def _insert_data(self, df_data: pd.DataFrame):
        sql = "INSERT INTO fund_data(fund_code, date, nav, c_nav, growth_rate) VALUES(%s, %s, %s, %s, %s);"
        data = df_data.values.tolist()
        try:
            with self.conn.cursor() as cur:
                cur.executemany(sql, data)
                self.conn.commit()
        except Error as e:
            logging.error(f"Error inserting data: {e}")
            for row in data:
                try:
                    cur.execute(sql, row)
                except Error as row_error:
                    logging.error(f"Error inserting row: {row} with error: {row_error}")

    def __del__(self):
        try:
            self.cursor.close()
            self.conn.close()
        except Exception as e:
            logging.error(f"Error closing connection: {e}")


if __name__ == "__main__":
    fund_codes = ["000016", "000033"]

    crawler = FundHistoryCrawl(max_requests_per_second=10)
    results = crawler.do_crawl(fund_codes)

    print(results)
    print("数据爬取完成！")
