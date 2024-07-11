import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.data import MyDataset
from src.lstm import LSTM
from src.utils import draw_fit_curve, de_normalization, eval_func


def predict(model: LSTM,
            fund_code: str,
            seq_len: int = 96,
            batch_size: int = 64,
            step: int = 1,
            predict_steps: int = 1  # 新增参数，用于控制在测试集耗尽后的预测步长
            ):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 测试集
    test_set = MyDataset(fund_code, seq_len, type_='test')
    test_loader = DataLoader(test_set, batch_size=batch_size, drop_last=True)

    # 模型
    print("---------开始预测---------")
    model = model.to(device)
    checkpoint = torch.load(f'checkpoints/{fund_code}.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    predictions = []  # 记录预测值
    groundtruths = []  # 记录真实值
    previous_output = None  # 用于记录上一步的预测值

    # 预测
    for seq, label in tqdm(test_loader, desc=f"预测中: {fund_code}"):
        seq, label = seq.to(device), label.to(device)
        output = model(seq)  # 前向传播
        predictions.append(output.cpu().detach().numpy())
        groundtruths.append(label.cpu().detach().numpy())
        previous_output = output

    # 测试集耗尽后，继续使用上一步的预测值进行预测
    if previous_output is not None:
        previous_output = previous_output.unsqueeze(0)  # 保持输入的批次维度
        for _ in tqdm(range(predict_steps), desc="继续预测"):
            output = model(previous_output)  # 使用上一步的预测值作为输入
            predictions.append(output.cpu().detach().numpy())
            previous_output = output.unsqueeze(0)  # 保持输入的批次维度

    print("---------预测完成---------")

    # tensor->numpy array
    predictions = np.concatenate(predictions, axis=0)
    groundtruths = np.concatenate(groundtruths, axis=0)

    # 评估
    eval_func(groundtruths, predictions)
    print("---------评估完成---------")

    # 反归一化
    predictions = de_normalization(predictions, test_set.scaler)
    groundtruths = de_normalization(groundtruths, test_set.scaler)

    return predictions, groundtruths


if __name__ == '__main__':
    pass
