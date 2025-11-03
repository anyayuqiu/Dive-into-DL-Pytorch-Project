import torch
import numpy as np

# Tensor转Numpy
a = torch.ones(5)
b = a.numpy()
print(a, b)

a += 1
print(a, b)

b += 1
print(a, b)

# NumPy转Tensor
a = np.ones(5)
b = torch.from_numpy(a)
print(a, b)

a += 1
print(a, b)

b += 1
print(a, b)

# 该方法不再共享内存
c = torch.tensor(a)
a += 1
print(a,c)

x = torch.rand(2,3)
if torch.cuda.is_available():
    device = torch.device("cuda")
    y = torch.ones_like(x, device=device)
    x = x.to(device)
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))
