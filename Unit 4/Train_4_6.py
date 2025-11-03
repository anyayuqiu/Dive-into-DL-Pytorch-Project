import torch
from torch import nn

print(torch.cuda.is_available())# 输出
print(torch.cuda.device_count())# 输出
print(torch.cuda.current_device()) # 输出 0
print(torch.cuda.get_device_name(0))

x = torch.tensor([1, 2, 3])
x = x.cuda(0)
print(x)
print(x.device)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# x = torch.tensor([1, 2, 3], device=device)
# # or
# x = torch.tensor([1, 2, 3]).to(device)
