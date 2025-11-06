import time
import torch
from torch import nn, optim
import torchvision
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 96, 11, 4),
        # in_channels, out_channels,kernel_size, stride, padding
                    nn.ReLU(),
                    nn.MaxPool2d(3, 2),
                    nn.Conv2d(96, 256, 5, 1, 2),
                    nn.ReLU(),
                    nn.MaxPool2d(3, 2),
        # 减小卷积窗口，使⽤用填充为2来使得输⼊入与输出的⾼高和宽⼀一致，且增⼤大输出通道数
                    nn.Conv2d(256, 384, 3, 1, 1),
                    nn.ReLU(),
                    nn.Conv2d(384, 384, 3, 1, 1),
                    nn.ReLU(),
                    nn.Conv2d(384, 256, 3, 1, 1),
                    nn.ReLU(),
                    nn.MaxPool2d(3, 2)
       )
        self.fc = nn.Sequential(
            nn.Linear(256 * 5 * 5, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            # 输出层。由于这⾥里里使用Fashion-MNIST，所以⽤用类别数为10，⽽而非论⽂文中的1000
        nn.Linear(4096, 10),
        )
    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output


net = AlexNet()
print(net)

batch_size = 128
# 如出现“out of memory”的报错信息，可减⼩小batch_size或resize
train_iter, test_iter = d2l.load_data_fashion_mnist1(batch_size,resize=224)

lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)