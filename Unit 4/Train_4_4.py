import torch
from torch import nn


#1 不含模型参数的自定义层

class CenteredLayer(nn.Module):
    def __init__(self, **kwargs):
        super(CenteredLayer,self).__init__(**kwargs)
    def forward(self, x):
        return x - x.mean()

layer = CenteredLayer()
out = layer(torch.tensor([1, 2, 3,  4, 5], dtype=float))


net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())

y = net(torch.randn(4, 8))
y.mean().item()

#2 含模型参数的自定义层
class MyListDense(nn.Module):
    def __init__(self):
        super(MyListDense, self).__init__()
        self.params = nn.ParameterList([nn.Parameter(torch.randn(4,4)) for i in range(3)])
        self.params.append(nn.Parameter(torch.randn(4, 1)))

    def forward(self, x):
        for i in range(len(self.params)):
            x = torch.mm(x, self.params[i])
        return x
net = MyListDense()
print(net)


class MyDictDense(nn.Module):
    def __init__(self):
        super(MyDictDense, self).__init__()
        self.params = nn.ParameterDict({
            'linear1': nn.Parameter(torch.randn(4, 4)),
            'linear2': nn.Parameter(torch.randn(4, 1))
        })
        self.params.update({'linear3': nn.Parameter(torch.randn(4,2))}) # 新增
    def forward(self, x, choice='linear1'):
        return torch.mm(x, self.params[choice])

net = MyDictDense()
print(net)

x = torch.ones(1, 4)
print(net(x, 'linear1'))
print(net(x, 'linear2'))
print(net(x, 'linear3'))

net = nn.Sequential(
 MyDictDense(),
 MyListDense(),
)
print(net)
print(net(x))