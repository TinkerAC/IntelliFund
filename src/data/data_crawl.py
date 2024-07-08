import requests
import json
from pymysql import connect
from time import sleep
import pandas as pd
from pandas import DataFrame


class PureFinance:
    def __init__(self, code):
        self.code = code
        self.conn = self._create_connection()

        print("开始初始化数据库表，删除该基金下错误数据...")
        with self.conn.cursor() as cur:
            self._delete_existing_data(cur)
            self._reset_auto_increment(cur)
        self.conn.commit()

    def _create_connection(self):
        return connect(host='localhost', port=3306, database='financedb', user='root', password='mysql')

    def _delete_existing_data(self, cur):
        sql_str = "DELETE FROM fund_history WHERE fund_code=%s;"
        row_count = cur.execute(sql_str, [self.code])
        print(f"表中受影响行数为 {row_count}")

    def _reset_auto_increment(self, cur):
        sql_str = "ALTER TABLE fund_history AUTO_INCREMENT = 1;"
        cur.execute(sql_str)

    def getNPV(self):
        """
        查询全部历史净值
        :return: 查询结果字典，成功或者失败
        """
        try:
            total_count = self._fetch_total_count()
            page_total = (total_count + 19) // 20  # 等价于 math.ceil(total_count / 20)
            print(f"总页数为 {page_total}")

            tmp_list = []
            for single_page in range(1, page_total + 1):
                print(f"现在处理第 {single_page} 页数据")
                page_data = self._fetch_page_data(single_page)
                tmp_list.extend(page_data)
                sleep(1)  # 避免过多请求

            self._insert_data(tmp_list)

            return {"message": "ok", "status": 200}
        except Exception as e:
            print(f"Error: {e}")
            return {"message": "error", "status": 400}
        finally:
            self.conn.close()

    def _fetch_total_count(self):
        url = f"http://api.fund.eastmoney.com/f10/lsjz?callback=jQuery18304038998523093684_1586160530315&fundCode={self.code}&pageIndex=1&pageSize=20"
        header = {"Referer": f"http://fundf10.eastmoney.com/jjjz_{self.code}.html"}
        response = requests.get(url, headers=header)
        response.raise_for_status()

        json_data = json.loads(response.text[41:-1])
        print(f"初次执行结果：\n{json_data}")
        return json_data.get("TotalCount")

    @staticmethod
    def _fetch_page_data(page, code) -> DataFrame:
        url = f"http://api.fund.eastmoney.com/f10/lsjz"
        params = {
            "callback": "jQuery18304038998523093684_1586160530315",
            "fundCode": code,
            "pageIndex": page,
            "pageSize": 20
        }
        header = {"Referer": f"http://fundf10.eastmoney.com/jjjz_{code}.html"}
        response = requests.get(url,
                                headers=header,
                                params=params)
        response.raise_for_status()

        json_data = json.loads(response.text[41:-1])
        list_date_data = json_data.get("Data", {"LSJZList": None}).get("LSJZList")
        return pd.DataFrame(list_date_data)

    def _insert_data(self, data):
        sql = "INSERT INTO fund_history(fund_code, date, NPV, rate) VALUES(%s, %s, %s, %s);"
        with self.conn.cursor() as cur:
            cur.executemany(sql, data)
            self.conn.commit()


if __name__ == "__main__":
    df = PureFinance._fetch_page_data(1, '970196')
    print(df.to_markdown())
