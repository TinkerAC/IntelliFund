def generate_sample(num):
    """生成样本数据
    """
    state = np.random.rand(state_size)  # 随机生成状态
    action = np.random.randint(0, action_size)  # 随机选择动作
    reward = np.random.rand()  # 随机生成奖励
    next_state = np.random.rand(state_size)  # 随机生成下一个状态
    done = np.random.choice([True, False])  # 随机决定是否结束
    return state, action, reward, next_state, done


def generate_samples(num):
    """生成多个样本数据
    """
    samples = [generate_sample(i) for i in range(num)]
    return samples
