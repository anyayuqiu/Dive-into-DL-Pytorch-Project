import torch

x = torch.tensor([1,2])
y = torch.tensor([3,4])

id_tmp = id(y)
# 该方式会开辟新的内存
y = y + x
print(id(y) == id_tmp)

x = torch.tensor([1,2])
y = torch.tensor([3,4])

id_tmp1 = id(y)

# 会写入原来y的地址
y[:] = y + x

print(id(y) == id_tmp1)
