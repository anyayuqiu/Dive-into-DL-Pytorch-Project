import torch
from torch import nn

def comp_conv2d(conv2d, x):
    # (1, 1)代表批量大小和通道数（“多输入通道和多输出通道”一节将介绍）均为1
    x = x.view((1,1) + x.shape)
    y = conv2d(x)
    return y.view(y.shape[2:]) # 排除不关心的前两维：批量和通道

# 注意这里是两侧分别填充1⾏行行或列，所以在两侧一共填充2行或列
conv2d = nn.Conv2d(in_channels=1, out_channels= 1, kernel_size=3, padding=1)

x = torch.rand(8,8)

comp_conv2d(conv2d,x).shape

conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
comp_conv2d(conv2d, x).shape