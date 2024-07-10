from src.models import LSTM
import copy
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.data import MyDataset
from src.utils import draw_loss_curve
import os


def train(fund_code='510050'
          , start_date='2020-01-01',
          end_date='2023-01-01',
          batch_size=64):
    input_size = 7  # 净值、累计净值、增长率、上证收盘价、上证交易量、深证收盘价、深证交易量
    output_size = 1  # 次日净值
    learning_rate = 1e-2
    seq_len = 96
    hidden_size = 96
    num_layers = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("-------数据集加载中-------")
    train_set = MyDataset(fund_code, start_date, end_date, seq_len, type_='train')
    valid_set = MyDataset(fund_code, start_date, end_date, seq_len, type_='valid')
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, drop_last=False)
    print("--------加载完成---------")

    print(len(train_set), len(valid_set))
    if len(train_loader) == 0 or len(valid_loader) == 0:
        print("数据集为空，请检查数据集")
        return

    print("----开始在" + str(device) + "上训练------")
    model = LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size).to(
        device)
    criterion = nn.MSELoss()  # 均方误差损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    num_epochs = 50
    train_losses = []
    valid_losses = []
    best_model = None
    min_valid_loss = float('inf')
    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_loss = 0
        for seq, label in train_loader:
            seq, label = seq.to(device), label.to(device)
            output = model(seq)  # 前向传播
            loss = criterion(output, label)  # 计算loss
            train_loss += loss.item()  # 累计每个批次的loss
            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # 计算梯度
            optimizer.step()  # 反向传播
        train_loss = train_loss / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        with torch.no_grad():
            valid_loss = 0
            for seq, label in valid_loader:
                seq, label = seq.to(device), label.to(device)
                output = model(seq)
                loss = criterion(output, label)
                valid_loss += loss.item()
            valid_loss = valid_loss / len(valid_loader)
            valid_losses.append(valid_loss)
            # 寻找在验证集上loss最小的模型
            if valid_loss < min_valid_loss:
                min_valid_loss = valid_loss
                best_model = copy.deepcopy(model)
    print("--------训练完成---------")

    draw_loss_curve(train_losses, valid_losses)
    print("-----loss曲线绘制完成-----")

    os.makedirs('checkpoints', exist_ok=True)
    torch.save({'model_state_dict': best_model.state_dict()}, f'checkpoints/{fund_code}.pt')
    print("-------模型保存完成-------")


if __name__ == '__main__':
    train()
