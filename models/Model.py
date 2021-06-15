import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

import numpy as np


def weight_init(layers):
    for layer in layers:
        if isinstance(layer, nn.BatchNorm1d):
            layer.weight.data.fill_(1)
            layer.bias.data.zero_()
        elif isinstance(layer, nn.Linear):
            n = layer.in_features
            y = 1.0 / np.sqrt(n)
            layer.weight.data.uniform_(-y, y)
            layer.bias.data.fill_(0)


class RNN(nn.Module):
    def __init__(self,
                 feature_nums,
                 hidden_dims,
                 bi_lstm,
                 out_dims=1):
        super(RNN, self).__init__()
        self.feature_nums = feature_nums # 输入数据特征维度
        self.hidden_dims = hidden_dims # 隐藏层维度
        self.bi_lism = bi_lstm # LSTM串联数量

        self.lstm = nn.LSTM(self.feature_nums, self.hidden_dims, self.bi_lism)
        self.out = nn.Linear(self.hidden_dims, out_dims)

    def forward(self,x):
        x1, _ = self.lstm(x)
        a, b, c = x1.shape
        out = self.out(x1.view(-1, c))
        out1 = out.view(a, b, -1)

        return out1
