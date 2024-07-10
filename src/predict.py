import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.data import MyDataset
from src.models import LSTM
from src.utils import draw_fit_curve, de_normalization, eval_func


def predict(fund_code, start_date, end_date):
    # 超参数

    batch_size = 64  # 批量大小
    input_size = 7  # 输入特征维度：净值、累计净值、增长率、上证收盘价、上证交易量、深证收盘价、深证交易量
    output_size = 1  # 输出特征维度：次日净值
    seq_len = 96  # 序列长度
    hidden_size = 96  # LSTM层神经元数量
    num_layers = 1  # LSTM层数
    device = torch.device("cpu")  # "cuda" if torch.cuda.is_available() else "cpu"

    # 测试集
    test_set = MyDataset(fund_code, start_date, end_date, seq_len, type_='test')
    test_loader = DataLoader(test_set, batch_size=batch_size, drop_last=True)

    # 模型
    print("---------开始预测---------")
    model = LSTM(input_size, hidden_size, num_layers, output_size).to(device)
    checkpoint = torch.load('checkpoints/best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    predictions = []  # 记录预测值
    groundtruths = []  # 记录真实值
    original_values = []  # 记录原始值
    # 预测
    for seq, label in tqdm(test_loader):
        seq, label = seq.to(device), label.to(device)
        output = model(seq)  # 前向传播
        predictions.append(output.cpu().detach().numpy())
        groundtruths.append(label.cpu().detach().numpy())
        original_values.append(seq.cpu().detach().numpy()[:, :, 0])  # 只记录净值

    print("---------预测完成---------")

    # tensor->numpy array
    predictions = np.concatenate(predictions, axis=0)
    groundtruths = np.concatenate(groundtruths, axis=0)
    original_values = np.concatenate(original_values, axis=0)

    # 评估
    eval_func(groundtruths, predictions)
    print("---------评估完成---------")

    # 反归一化
    predictions = de_normalization(predictions, test_set.scaler)
    groundtruths = de_normalization(groundtruths, test_set.scaler)

    # 绘制拟合曲线
    draw_fit_curve(predictions, groundtruths, fund_code=fund_code)
    print("------拟合曲线绘制完成------")


if __name__ == '__main__':
    eval()
