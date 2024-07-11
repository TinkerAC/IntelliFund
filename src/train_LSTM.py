from src.lstm import LSTM
import copy
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.data import MyDataset
from src.utils import draw_loss_curve
import os
import traceback
import math


def train(
        model: LSTM,
        fund_code='510050',
        batch_size=64,
        num_epochs=50,
        learning_rate=1e-2,
        seq_len: int = 96,
        data_set_length: int = None
):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("-------数据集加载中-------")
        train_set = MyDataset(fund_code, seq_len, type_='train', data_set_length=data_set_length)
        valid_set = MyDataset(fund_code, seq_len, type_='valid', data_set_length=data_set_length)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
        valid_loader = DataLoader(valid_set, batch_size=batch_size, drop_last=False)
        print("--------加载完成---------")

        assert len(train_set) > 0, "训练集为空"

        print("----开始在" + str(device) + "上训练------")
        model = model.to(device)
        criterion = nn.MSELoss()  # 均方误差损失函数
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        train_losses = []
        valid_losses = []
        best_model = copy.deepcopy(model)  # 初始化 best_model
        min_valid_loss = float('inf')

        for epoch in tqdm(range(num_epochs),
                          desc=f"训练中: {fund_code}"):
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
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        traceback.print_exc()


if __name__ == '__main__':
    pass
