import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

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
        out = self.fc(out[:, -1, :])

        return out

    def __str__(self):
        return 'LSTM(input_size=%d, hidden_size=%d, num_layers=%d, output_size=%d)' % (
            self.input_size, self.hidden_size, self.num_layers, self.output_size)


# 示例用法
if __name__ == '__main__':
    input_tensor = torch.rand(32, 96, 7)  # 32批量，96的长度，7个特征
    model = LSTM(input_size=7, hidden_size=96, num_layers=1, output_size=1)
    output = model(input_tensor)
    print(output.shape)
    print(model)
