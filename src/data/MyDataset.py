import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy import create_engine
import pathlib
from src.utils import normalization, de_normalization


class MyDataset(Dataset):
    def __init__(self, fund_code: str,
                 start_date: str,
                 end_date: str,
                 seq_len: int,
                 type_: str,
                 max_duration: bool = False,
                 sh_index_path: str = str(pathlib.Path(__file__).parent.parent.parent / 'data' / 'raw' / 'sh.csv'),
                 sz_index_path: str = str(pathlib.Path(__file__).parent.parent.parent / 'data' / 'raw' / 'sz.csv')):
        self.seq_len = seq_len

        # 加载数据
        engine = create_engine('mysql+pymysql://root:mysql@localhost:3306/financedb')
        if max_duration:
            query = f'SELECT * FROM fund_data WHERE fund_code = "{fund_code}"'
        else:
            query = f'SELECT * FROM fund_data WHERE fund_code = "{fund_code}" AND date >= "{start_date}" AND date <= "{end_date}"'
        fund_data = pd.read_sql(query, con=engine)

        sh_index_data = pd.read_csv(sh_index_path)
        sz_index_data = pd.read_csv(sz_index_path)

        # 数据对齐
        fund_data['date'] = pd.to_datetime(fund_data['date'])
        sh_index_data['Date'] = pd.to_datetime(sh_index_data['Date'])
        sz_index_data['Date'] = pd.to_datetime(sz_index_data['Date'])

        # 重命名日期列以便合并
        sh_index_data.rename(columns={'Date': 'date'}, inplace=True)
        sz_index_data.rename(columns={'Date': 'date'}, inplace=True)

        # 合并数据
        merged_data = pd.merge(fund_data, sh_index_data[['date', 'Close', 'Volume']], on='date', how='inner')
        merged_data = pd.merge(merged_data, sz_index_data[['date', 'Close', 'Volume']], on='date', how='inner',
                               suffixes=('_sh', '_sz'))

        # 选择特征
        self.data = merged_data[['nav', 'c_nav', 'growth_rate', 'Close_sh', 'Volume_sh', 'Close_sz', 'Volume_sz']]

        # 归一化
        self.data, self.scaler = normalization(self.data)

        # 确保划分数据集时考虑到 seq_len 的影响
        total_len = len(self.data) - self.seq_len
        train_end = int(total_len * 0.7)
        valid_end = int(total_len * 0.85)

        if type_ == 'train':
            self.data = self.data[:train_end + self.seq_len]
        elif type_ == 'valid':
            self.data = self.data[train_end:valid_end + self.seq_len]
        elif type_ == 'test':
            self.data = self.data[valid_end:]

        # 打印调试信息
        print(f"Dataset type: {type_}, data length: {len(self.data)}")

        self.data = torch.tensor(self.data, dtype=torch.float32)

    def __getitem__(self, idx):
        seq = self.data[idx:idx + self.seq_len, :]
        label = self.data[idx + self.seq_len, 0]
        return seq, label

    def __len__(self):
        return len(self.data) - self.seq_len


if __name__ == '__main__':
    train_dataset = MyDataset(fund_code='510050',
                              start_date='2020-01-01',
                              end_date='2023-01-01',
                              seq_len=96,
                              type_='train')

    valid_dataset = MyDataset(fund_code='510050',
                              start_date='2020-01-01',
                              end_date='2023-01-01',
                              seq_len=96,
                              type_='valid')

    test_dataset = MyDataset(fund_code='510050',
                             start_date='2020-01-01',
                             end_date='2023-01-01',
                             seq_len=96,
                             type_='test')

    loader1 = DataLoader(train_dataset, batch_size=32, shuffle=True)
    loader2 = DataLoader(valid_dataset, batch_size=32, shuffle=True)
    loader3 = DataLoader(test_dataset, batch_size=32, shuffle=True)

    for seq, label in loader1:
        print(f"Train batch - seq shape: {seq.shape}, label shape: {label.shape}")
        break

    for seq, label in loader2:
        print(f"Valid batch - seq shape: {seq.shape}, label shape: {label.shape}")
        break

    for seq, label in loader3:
        print(f"Test batch - seq shape: {seq.shape}, label shape: {label.shape}")
        break
