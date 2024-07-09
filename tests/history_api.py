import re
import json

import pandas as pd
import requests
from pprint import pprint


def test_history_api(fund_code='161603'):
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
    return pd.DataFrame(json.loads(re.findall(r'\((.*)\)', response.text)[0]).get('Data', {}).get('LSJZList', {}))


if __name__ == '__main__':
    pass
