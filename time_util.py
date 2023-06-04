import torch
import numpy as np
import torch.nn as nn
import pandas as pd


# 时间差编码函数
class DeltaEncoder(torch.nn.Module):
    def __init__(self, dimension):
        super(DeltaEncoder, self).__init__()

        self.dimension = dimension
        self.w = torch.nn.Linear(1, dimension)

        self.w.weight = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dimension)))
                                           .float().reshape(dimension, -1))
        self.w.bias = torch.nn.Parameter(torch.zeros(dimension).float())

    def forward(self, t):
        """
        输入时间差序列，输入时间差表征
        @param t: 输入的时间差,格式为torch.Tensor,形状为(batch)
        @return: 时间差表征，形状为(batch,dim)
        """
        t = t.unsqueeze(dim=-1)

        # output has shape [batch_size, dimension]
        output = torch.cos(self.w(t))
        return output


# Semantic时间编码函数
class SemanticEncoder(torch.nn.Module):
    # Time Encoding proposed by TGAT
    def __init__(self, dimension, device):
        super(SemanticEncoder, self).__init__()
        self.dimension = dimension
        self.week_emb = nn.Embedding(7, dimension)
        self.month_emb = nn.Embedding(12, dimension)
        self.day_emb = nn.Embedding(31, dimension)
        self.hour_emb = nn.Embedding(24, dimension)
        self.device = device

    def forward(self, t):
        """
        输入时间戳序列，输出时间戳表征
        @param t: 输入的时间差,格式为 torch.Tensor,形状为(batch)
        @return: 时间差表征，形状为(batch,dim)
         """
        #new_t = t.cpu().numpy()
        # 将unix时间戳转化为时间,其他时间格式可以用其他转换函数
        new_t = pd.to_datetime(t, unit='s')
        # print("new_t:", new_t)
        # 获取时间戳的星期，天，月份，小时，可自行修改
        week, day, month, hour = new_t.weekday.values, new_t.day.values, new_t.month.values, new_t.hour.values
        week_emb = self.week_emb(torch.tensor(week, device=self.device))
        day_emb = self.day_emb(torch.tensor(day - 1, device=self.device))
        month_emb = self.month_emb(torch.tensor(month - 1, device=self.device))
        hour_emb = self.hour_emb(torch.tensor(hour, device=self.device))
        return hour_emb


#Test
# model = SemanticEncoder(5, torch.device('cpu'))
# a = pd.to_datetime(['2020-01-03 13:45:11', '2020-01-04 15:55:12']).values.astype(np.int64) // 10 ** 9
# x = model.forward(torch.tensor(a))
