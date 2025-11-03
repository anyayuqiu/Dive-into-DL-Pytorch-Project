import torch
from torch import nn



#1 继承 MODULE 类来构造模型
class MLP(nn.Module):
    #声明带有模型参数的层，这⾥里声明了了两个全连接层
    def __init__(self, **kwargs):
        # 调⽤用MLP⽗父类Block的构造函数来进⾏行行必要的初始化。这样在构造实例例时还可 以指定其函数
        # 参数，如“模型参数的访问、初始化和共享”⼀一节将介绍的模型参数params
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Linear(784,256)
        self.act = nn.ReLU()
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)

x = torch.rand(2, 784)
net = MLP()
print("net1:",net)




#2 MODULE 的⼦子类 - Sequential 类
class MySequential(nn.Module):
    from collections import OrderedDict
    def __init__(self, *args):
        super(MySequential,self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def forward(self, input):
        for module in self._modules.values():
            input = module(input)
        return input

net = MySequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)
print("net2:",net)

#3 MODULE 的子类 - ModuleList 类
#ModuleList 接收一个子模块的列表作为输⼊入，然后也可以类似List那样进⾏行行append和extend操作
net = nn.ModuleList([nn.Linear(784, 256), nn.ReLU()])
net.append(nn.Linear(256,10))
print("net3:",net[-1])
print("net3:",net)

#4 MODULE 的子类 - ModuleDict 类
net = nn.ModuleDict({
    'linear':nn.Linear(784, 256),
    'act':nn.ReLU(),
})
net['output'] = nn.Linear(256, 10)
print("net4:",net['linear'])

#5 构造复杂网络模型
class FancyMLP(nn.Module):
    def __init__(self, **kwargs):
        super(FancyMLP, self).__init__(**kwargs)

        self.rand_weight = torch.rand((20,20), requires_grad=False) # 常数参数
        self.linear = nn.Linear(20,20)

        def forward(self, x):
            x = self.linear(x)
            # 使用创建的常数参数，以及nn.functional中的relu函数和mm函数
            x = nn.functional.relu(torch.mm(x, self.rand_weight.data) + 1)

            # 复用全连接层。等价于两个全连接层共享参数
            x = self.linear(x)

            # 控制流，这⾥里里我们需要调⽤用item函数来返回标量量进⾏行行⽐比较
            while x.norm().item() > 1:
                x /= 2
            if x.norm().item() < 0.8:
                x *= 10
            return x.sum()


class NestMLP(nn.Module):
    def __init__(self, **kwargs):
        super(NestMLP, self).__init__(**kwargs)
        self.net = nn.Sequential(nn.Linear(40, 30), nn.ReLU())

    def forward(self, x):
        return self.net(x)


net = nn.Sequential(NestMLP(), nn.Linear(30, 20), FancyMLP())
X = torch.rand(2, 40)
print("net5:",net)