import os

from src import train, predict

if __name__ == '__main__':
    fund_code = '217003'
    start_date = '2021-01-01'
    end_date = '2024-01-01'

    train(
        fund_code=fund_code,
        start_date=start_date,
        end_date=end_date
    )
    predict(
        fund_code=fund_code,
        start_date=start_date,
        end_date=end_date)
