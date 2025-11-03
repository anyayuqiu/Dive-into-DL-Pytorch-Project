import torch

#未初始化创建
x = torch.empty(5, 3)
print(x)

#随机初始化
x = torch.rand(5, 3)
print(x)

#控制类型创建
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

#直接赋值创建
x = torch.tensor([1,3])
print(x)

x = torch.tensor([[1.1, 2],[3,4.4]])
print(x)

#修改tensor对象的数据类型
x = x.new_ones(5,3, dtype=torch.float64)
print(x)

#randn_like(x,dtype = torch.float) 返回维度同x的随机张量
x = torch.randn_like(x, dtype=torch.float)

print(x.size())
print(x.shape)

#生成在区间 [0, 1)内均匀分布的随机数
x = torch.rand(2,2)
print(x)

#生成均值为0， 标准差为1的标准正态分布
x = torch.randn(2,2)
print(x)

#torch.normal(a,b)生成均值为a,标准差为b的正态分布
x = torch.normal(1,1,size=(2,3))
print(x)

#torch.uniform(a,b)生成[a,b)的均匀分布数
x = torch.zeros(2,3)
x.uniform_(1,5)
print(x)


