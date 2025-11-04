import torch
from torch import nn
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l


#1 互相关运算
x = torch.tensor([[0,1,2],[3,4,5]])
k = torch.tensor([[0,1],[2,3]])

y = d2l.corr2d(x,k)

#print(y)

#2 二维卷积层
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super(Conv2D, self).__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return d2l.corr2d(x,self.weight) + self.bias


#3 图像物体边缘检测

x = torch.ones(6,8)
x[:,2:6] = 0
print("x:",x)
k = torch.tensor([[1, -1]])
y = d2l.corr2d(x, k)
print("y:",y)


#4 通过数据学习核数组

conv2d = Conv2D(kernel_size=(1,2))

step = 20
lr = 0.01
for i in range(step):
    y_hat = conv2d(x)
    l = ((y_hat - y) ** 2).sum()
    l.backward()

    conv2d.weight.data -= lr * conv2d.weight.grad
    conv2d.bias.data -= lr * conv2d.bias.grad

    conv2d.weight.grad.fill_(0)
    conv2d.bias.grad.fill_(0)
    if (i + 1) % 5 == 0:
        print('Step %d, loss %.3f' % (i+1, l.item()))

print("weight: ", conv2d.weight.data)
print("bias: ", conv2d.bias.data)

