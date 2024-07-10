from sqlalchemy import create_engine


def create_my_engine():
    engine = create_engine('mysql+pymysql://root:mysql@localhost:3306/financedb')
    return engine
