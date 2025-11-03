import torch

#加法操作
x = torch.empty(2,3)
x.normal_(1,5)
y = torch.rand(2,3)

#1
print(x+y)

#2
print(torch.add(x,y))

#指定输出
result = torch.empty(2,3)
torch.add(x, y, out = result)
print(result)

#3
y.add_(x)
print(y)

#类似取地址
y = x[0, :]
y += 1
print(y)
print(x[0, :]) # 源tensor也被改了

#改变形状 view()返回的新tensor与源tensor共享内存（其实是同⼀一个tensor）
y = x.view(6)
z = x.view(-1, 1) # -1所指的维度可以根据其他维度的值推出来
print(x.size(), y.size(), z.size())

#返回新的副本
x_copy = x.clone().view(6)
x -= 1
print(x)
print(x_copy)

#将tensor转化为Python number
x = torch.randn(1)
print(x)
print(x.item())


