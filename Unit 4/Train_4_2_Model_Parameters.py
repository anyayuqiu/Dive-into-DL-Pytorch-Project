import torch
from torch import nn
from torch.nn import init

#1 访问模型参数
net = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 1)) #pytorch已进行行默认初始化
print(net)


print(type(net.named_parameters()))
for name, param in net[0].named_parameters():
    print(name, param.size(), type(param))


class MyModel(nn.Module):
    def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self.weight1 = nn.Parameter(torch.rand(20, 20))
        self.weight2 = torch.rand(20, 20)

    def forward(self, x):
        pass


n = MyModel()
for name, param in n.named_parameters():
    print(name)

#2 初始化模型参数
for name, param in net.named_parameters():
    if 'weight' in name:
        init.normal_(param, mean=0, std=0.01)
        print(name, param.data)

for name, param in net.named_parameters():
    if 'bias' in name:
        init.constant_(param, val=0)
        print(name, param.data)

#3 自定义初始化方法



def normal_(tensor, mean=0, std=1):
    with torch.no_grad():
        return tensor.normal_(mean, std)

def init_weight_(tensor):
    with torch.no_grad():
        tensor.uniform_(-10, 10)
        tensor *= (tensor.abs() >= 5).float()

for name, param in net.named_parameters():
    if 'weight' in name:
        init_weight_(param)
        print(name, param.data)

for name, param in net.named_parameters():
    if 'bias' in name:
        param.data += 1
        print(name, param.data)


linear = nn.Linear(1, 1, bias=False)
net = nn.Sequential(linear, linear)
print(net)
for name, param in net.named_parameters():
    init.constant_(param, val=3)
    print(name, param.data)

x = torch.ones(1, 1)
y = net(x).sum()
print(y)
y.backward()
print(net[0].weight.grad)
