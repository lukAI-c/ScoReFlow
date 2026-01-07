import torch
from torch import nn

class RunningMeanStd(nn.Module):
    """
    动态计算数据的均值和方差 (Welford's algorithm 的批量版本)
    继承 nn.Module 主要是为了方便处理 device 和 state_dict
    """
    def __init__(self, epsilon=1e-4, shape=()):
        super().__init__()
        # 使用 register_buffer 确保这些参数会被保存到 checkpoint，但不会被 optimizer 更新
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("var", torch.ones(shape))
        self.register_buffer("count", torch.tensor(epsilon))

    def update(self, x):
        """更新统计量"""
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0, unbiased=False)
        batch_count = x.shape[0]
        
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + torch.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x):
        """将 x 归一化到标准正态分布"""
        return (x - self.mean) / torch.sqrt(self.var + 1e-8)

    def denormalize(self, x):
        """将标准化的 x 还原回真实尺度"""
        return x * torch.sqrt(self.var + 1e-8) + self.mean