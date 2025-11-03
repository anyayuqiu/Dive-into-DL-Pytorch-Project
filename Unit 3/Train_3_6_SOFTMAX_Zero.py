import torch
import torchvision
import numpy as np
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

#1 获取数据集加载器
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs = 28 * 28
num_outputs = 10

#2 初始化模型参数
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), requires_grad=True,dtype=torch.float)
b = torch.zeros(num_outputs, requires_grad=True, dtype=torch.float)

#3 定义SOFTMAX运算
# # dim = 0 按列操作, dim = 1 按行操作
# x = torch.tensor([[1, 2, 3], [4, 5, 6]])
# print(x.sum(dim=0, keepdim=True))
# print(x.sum(dim=1, keepdim=True))

def softmax(x):
    x_exp = x.exp()
    partition = x_exp.sum(dim = 1, keepdim = True)
    return x_exp / partition

#4 定义模型
def net(x):
    return softmax(torch.mm(x.view(-1, num_inputs), w) + b)

#5 定义损失函数
##gather(1, y.view(-1, 1))表示dim = 1,按行操作，取第0行第0个和第1行第2个 故结果是[[0.1],[0.5]]
# y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
# y = torch.LongTensor([0, 2])
# y_hat.gather(1, y.view(-1, 1))

def cross_entropy(y_hat, y):
    return - torch.log(y_hat.gather(1, y.view(-1,1))).sum()

#6 定义计算准确值(评价函数)
def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()

#7 训练模型
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y)

            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            if optimizer is None:
                d2l.sgd(params, lr, batch_size)
            else:
                optimizer.step()

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).float().mean().item()
            n += len(y)

        test_acc = d2l.evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f' % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))


num_epochs, lr = 20, 0.5
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [w,b], lr)