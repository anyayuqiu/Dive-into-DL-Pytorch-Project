import torch
from torch import nn
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l


#1 多输入通道
def corr2d_multi_in(X, K):
    # 沿着X和K的第0维（通道维）分别计算再相加
    res = d2l.corr2d(X[0, :, :], K[0, :, :])
    for i in range(1, X.shape[0]):
        res += d2l.corr2d(X[i, :, :], K[i, :, :])
    return res


X = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                  [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
K = torch.tensor([[[0, 1],
                   [2, 3]],
                  [[1, 2],
                   [3, 4]]])
corr2d_multi_in(X, K)

#2 多输出通道
def corr2d_multi_in_out(X, K):
    # 对K的第0维遍历，每次同输⼊入X做互相关计算。所有结果使⽤用stack函数合并在⼀一起
    return torch.stack([corr2d_multi_in(X, k) for k in K])

#构建 3 * 2 * 2 * 2
K = torch.stack([K, K + 1, K + 2])
# print(K.shape)
# print(corr2d_multi_in_out(X, K))

def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.view(c_i, h * w)
    K = K.view(c_o, c_i)
    Y = torch.mm(K, X) # 全连接层的矩阵乘法
    return Y.view(c_o, h, w)

#经验证，做 卷积时，以上函数与之前实现的互相关运算函数 corr2d_multi_in_out 等价。
X = torch.rand(3, 3, 3) # Ci * H * W
K = torch.rand(2, 3, 1, 1) # Co * Ci * Hk * Wk
Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
print((Y1 - Y2).norm().item() < 1e-6)