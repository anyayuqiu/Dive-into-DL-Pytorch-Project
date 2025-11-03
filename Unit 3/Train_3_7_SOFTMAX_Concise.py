import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

#1 获取数据加载器
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

#2 定义和初始化模型
num_inputs = 28 * 28
num_outputs = 10

class LinearNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(num_inputs,num_outputs)
    def forward(self, x):
        y = self.linear(x.view(-1, num_inputs))
        return y

net = LinearNet(num_inputs, num_outputs)
from collections import OrderedDict

net = nn.Sequential(
    OrderedDict([
        ('flatten', d2l.FlattenLayer()),
        ('linear', nn.Linear(num_inputs,num_outputs))
    ])
)
init.normal_(net.linear.weight, mean=0, std=0.1)
init.constant_(net.linear.bias, val=0)

#3 定义损失函数
loss = nn.CrossEntropyLoss()

#4 定义优化算法
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

#5 训练模型
num_epochs = 5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)

