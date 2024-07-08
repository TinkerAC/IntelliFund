import re
import json
import requests
from pprint import pprint


def history_api(fund_code='001938'):
    url = r"http://api.fund.eastmoney.com/f10/lsjz"
    params = {
        'callback': 'jQuery18304038998523093684_1586160530315',
        'fundCode': fund_code,
        'pageIndex': 1,
        'pageSize': 20
    }
    headers = {
        'Referer': f'http://fundf10.eastmoney.com/jjjz_{fund_code}.html'
    }

    response = requests.get(
        url=url,
        params=params,
        headers=headers
    )
    print(response.text)
    return response


if __name__ == '__main__':
    response = history_api()
    data_dict = json.loads(re.findall(r'\((.*)\)', response.text)[0])
    pprint(data_dict)
