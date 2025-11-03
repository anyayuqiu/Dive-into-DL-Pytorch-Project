import torch
import numpy as np
import random

from torch import nn

#1 生成数据集
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2

features = torch.from_numpy(np.random.normal(0, 1, (num_examples, num_inputs))).float()
labels = true_w[0] * features[:,0] + true_w[1] * features[:,1] + true_b
labels += torch.from_numpy(np.random.normal(0, 0.01, size = labels.size())).float()

#2 读取数据
import torch.utils.data as Data

batch_size = 10
dataset = Data.TensorDataset(features, labels)
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)

#3 定义模型
class LinearNet(nn.Module):
    #定义初始化
    def __init__(self, nums_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(nums_feature,1)
    #定义前向传播
    def forward(self, x):
        y = self.linear(x)
        return y

net = LinearNet(num_inputs)
print(net)

# # 写法⼀
# net = nn.Sequential(
#  nn.Linear(num_inputs, 1)
#  # 此处还可以传⼊入其他层
#  )
# # 写法⼆
# net = nn.Sequential()
# net.add_module('linear', nn.Linear(num_inputs, 1))
# # net.add_module ......
# # 写法三
# from collections import OrderedDict
# net = nn.Sequential(OrderedDict([
#  ('linear', nn.Linear(num_inputs, 1))
#  # ......
#  ]))
# print(net)
# print(net[0])

#4 初始化模型参数
from torch.nn import init

init.normal_(net.linear.weight, mean = 0, std = 0.01)
init.constant_(net.linear.bias, val = 0)

#5 定义损失函数
loss = nn.MSELoss()

#6 定义迭代算法
import torch.optim as optim

optimizer = optim.SGD(net.parameters(),lr = 0.03)

#7 训练模型
num_epochs = 10
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(output.size()))
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))

dense = net.linear
print(true_w, dense.weight)
print(true_b, dense.bias)