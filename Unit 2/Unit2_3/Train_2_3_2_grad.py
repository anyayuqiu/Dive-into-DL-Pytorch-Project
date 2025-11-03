import torch

#x为直接创造 为叶子节点
x = torch.ones(2, 2, requires_grad=True)
print(x)
print(x.grad_fn)

y = x + 2
print(y.grad_fn)

print(x.is_leaf, y.is_leaf)

z = y * y * 3
out = z.mean()
print(z, out)

#默认情况为requires为false
a = torch.randn(2,2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
b = (a * a).sum()
print(b.grad_fn)

#out 使用反向传播 计算关于 x 的梯度
out.backward()
print(x.grad)

#x.grad 是会累加的，所以在反向传播前应该清零
out1 = x.sum()
out1.backward()
print(x.grad)

# 梯度清零
out2 = x.sum()
x.grad.data.zero_()
out2.backward()
print(x.grad)

#我们只允许标量对张量求导，故对于张量z，引入权重矩阵进行加权求和转化为标量
x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
y = 2 * x
z = y.view(2,2)
print(z)
z.backward(torch.tensor([[1,2],[3,4]],dtype=torch.float))
print(x.grad)

#梯度中断
x = torch.tensor(1.0, requires_grad=True)
y1 = x ** 2
with torch.no_grad():
 y2 = x ** 3
y3 = y1 + y2

print(x.requires_grad)
print(y1, y1.requires_grad) # True
print(y2, y2.requires_grad) # False
print(y3, y3.requires_grad) # True

y3.backward()
#因为y2的梯度不会回传 故y3的梯度只y1相关
print(x.grad)

#修改tensor的数值不被autograd记录
x = torch.ones(1, requires_grad = True)
print(x.data)
print(x.requires_grad)

y = 2 * x
x.data *= 100


y.backward()
print(x) #[100]
print(x.grad)[2]

