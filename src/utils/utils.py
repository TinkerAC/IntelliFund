import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from pylab import mpl


# 工具箱，定义各种使用函数
def load_data(path):  # 数据加载函数
    """读取负荷数据"""
    data = pd.read_excel(path)  # 导入数据
    data = data.iloc[:, 3:11]  # 选取负荷和温度序列
    return data


def split_dataset(data, type):  # 区域划分函数
    """划分数据集"""
    if type == 'train':
        data = data[:int(data.shape[0] * 0.6), :]  # 训练集
    elif type == 'valid':
        data = data[int(data.shape[0] * 0.6):int(data.shape[0] * 0.9), :]  # 验证集
    elif type == 'test':
        data = data[int(data.shape[0] * 0.9):, :]  # 测试集
    return data


def normalization(data) -> (np.ndarray, MinMaxScaler):
    """归一化"""
    scaler = MinMaxScaler()
    norm_data = scaler.fit_transform(data)
    return norm_data, scaler


def de_normalization(predictions: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    """
    反归一化预测结果

    参数:
    predictions (np.ndarray): 预测的净值数据
    scaler (MinMaxScaler): 用于反归一化的缩放器

    返回:
    np.ndarray: 反归一化后的预测净值数据
    """
    # 使用 scaler 的 min 和 scale 来反归一化
    return predictions * (scaler.data_max_[0] - scaler.data_min_[0]) + scaler.data_min_[0]



def draw_loss_curve(train_losses, valid_losses):
    """绘制loss曲线"""
    epoches = list(range(1, len(train_losses) + 1))  # 设为小时为单位
    mpl.rcParams["font.sans-serif"] = ["SimHei"]  # 设置显示中文字体
    mpl.rcParams["axes.unicode_minus"] = False  # 设置正常显示符号
    plt.plot(epoches, train_losses, label='train_loss')  # 绘制train_loss线
    plt.plot(epoches, valid_losses, label='valid_loss')  # 绘制valid_loss线
    plt.xlabel('Epoch')  # 命名x轴
    plt.ylabel('Loss')  # 命名y轴
    plt.title('Loss曲线')  # 命名标题
    plt.legend(loc='upper right')  # 绘制图例
    plt.savefig('figures/Loss曲线.png')  # 保存
    plt.close()


def draw_fit_curve(predictions, grandtruths, fund_code,type='基金净值'):
    """
    绘制拟合曲线
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示问题

    plt.figure(figsize=(10, 5))
    time_steps = range(len(predictions))

    # 打印数据的维度以进行调试
    print("Predictions shape:", np.array(predictions).shape)
    print("Grandtruths shape:", np.array(grandtruths).shape)
    print("Time steps shape:", np.array(time_steps).shape)

    plt.plot(time_steps, grandtruths, 'g.--', label='真实值')  # 绘制真实值曲线
    plt.plot(time_steps, predictions, 'r.-', label='预测值')  # 绘制预测值曲线
    plt.xlabel('Time Step')
    plt.ylabel(type)
    plt.legend()
    plt.title(f'基金{fund_code}净值 预测值 vs 真实值')
    plt.show()


def eval_func(actual_y, forecast_y):
    """评估模型"""
    mse = np.mean((forecast_y - actual_y) ** 2)  # MSE
    mae = np.mean(np.abs(forecast_y - actual_y))  # MAE
    rmse = np.sqrt(np.mean((forecast_y - actual_y) ** 2))  # RMSE
    smape = np.mean(np.abs(forecast_y - actual_y) / (0.5 * (np.abs(actual_y) + np.abs(forecast_y)))) * 100  # SMAPE

    # 保存结果
    np.savetxt('results/mse.txt', mse.reshape(-1), fmt='%.4f')
    np.savetxt('results/mae.txt', mae.reshape(-1), fmt='%.4f')
    np.savetxt('results/rmse.txt', rmse.reshape(-1), fmt='%.4f')
    np.savetxt('results/smape.txt', smape.reshape(-1), fmt='%.4f')


"""
if __name__ == '__main__':
    # 测试
    train_loss_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    valid_loss_list = [0.2, 0.4, 0.6, 0.8, 1.0]
    epoch_list = [1, 2, 3, 4, 5]
    draw_loss_curve(train_loss_list, valid_loss_list)
"""
