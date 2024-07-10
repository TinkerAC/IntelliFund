import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 定义LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # 定义全连接层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # 前向传播LSTM层
        out, _ = self.lstm(x, (h_0, c_0))  # out: (batch_size, seq_len, hidden_size)

        # 取最后一个时间步的输出并通过全连接层
        out = self.fc(out[:, -1, :])  # out: (batch_size, output_size)

        return out

    def __str__(self):
        return 'LSTM(input_size=%d, hidden_size=%d, num_layers=%d, output_size=%d)' % (
            self.lstm.input_size, self.lstm.hidden_size, self.lstm.num_layers, self.fc.out_features)


if __name__ == '__main__':
    # 示例模型初始化
    input_size = 7  # 输入特征维度：净值、累计净值、增长率、上证收盘价、上证交易量、深证收盘价、深证交易量
    hidden_size = 96  # LSTM层神经元数量
    num_layers = 1  # LSTM层数
    output_size = 1  # 输出特征维度：次日净值

    model = LSTM(input_size, hidden_size, num_layers, output_size)
    print(model)
