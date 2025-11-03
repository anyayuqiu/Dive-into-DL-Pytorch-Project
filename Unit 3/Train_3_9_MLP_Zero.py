import numpy.random
import torch
import numpy as np
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

#1 获取和读取数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

#2 定义模型
num_inputs, num_outputs, num_hiddens = 28 * 28, 10, 256

w1 = torch.tensor(np.random.normal(0,0.01, (num_inputs,num_hiddens)), dtype=torch.float)
b1 = torch.zeros(num_hiddens, dtype=torch.float)

w2 = torch.tensor(numpy.random.normal(0, 0.01, (num_hiddens,num_outputs)), dtype=torch.float)
b2 = torch.zeros(num_outputs, dtype=torch.float)

params = [w1, b1, w2, b2]
for param in params:
    param.requires_grad_(requires_grad=True)


#3 定义激活函数
def relu(x):
    return torch.max(input=x, other=torch.tensor(0.0))

#4 定义模型
def net(x):
    x = x.view(-1, num_inputs)
    H = relu(torch.mm(x,w1) + b1)
    return torch.mm(H,w2)+b2

#5 定义损失函数 CrossEntropyLoss()带softmax和交叉熵
loss = torch.nn.CrossEntropyLoss()

#6 训练模型
num_epochs, lr = 10, 100
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)

