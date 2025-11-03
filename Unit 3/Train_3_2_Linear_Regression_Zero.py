import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random

#num

num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2

features = torch.from_numpy(np.random.normal(0, 1, (num_examples, num_inputs))).float()
labels = true_w[0] * features[:,0] + true_w[1] * features[:,1] + true_b
labels += torch.from_numpy(np.random.normal(0, 0.01, size = labels.size())).float()
print(features[0],labels[0])


def set_figsize(figsize=(3.5,2.5)):
    plt.rcParams['figure.figsize'] = figsize
    plt.rcParams['savefig.format'] = 'svg'

set_figsize()

plt.scatter(features[:, 0].numpy(), labels.numpy(),1)
#plt.show()

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i+batch_size,num_examples)])
        yield  features.index_select(0,j), labels.index_select(0,j)

batch_size = 10

# for X,y in data_iter(batch_size, features, labels):
#     print(X,y)
#     break



#初始化模型参数
w = torch.tensor(np.random.normal(0, 0.01,(num_inputs,1)), dtype = torch.float32)
b = torch.zeros(1, dtype = torch.float32)

w.requires_grad_(True)
b.requires_grad_(True)

#定义模型
def linreg(X, w, b):
    return torch.mm(X, w) + b

#定义损失函数
def squard_loss(y_hat, y):
    print(y_hat.size())
    print(y.size())
    return (y_hat - y.view(y_hat.size())) ** 2 / 2

#定义优化算法
def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size

##训练模型

lr = 0.03
num_epochs = 3
net = linreg
loss = squard_loss

for num_epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y).sum()

        l.backward()

        sgd([w, b], lr, batch_size)

        w.grad.data.zero_()
        b.grad.data.zero_()

    train_l = loss(net(features, w, b),labels)
    print('epoch %d, loss %f' % (num_epoch + 1, train_l.mean().item()))
