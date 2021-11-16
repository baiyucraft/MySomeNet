import torch
from torch import nn


class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BaseConv, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Bottle(nn.Module):
    def __init__(self, channels):
        super(Bottle, self).__init__()
        ex_channels = channels // 2
        self.base1 = BaseConv(channels, ex_channels, kernel_size=1)
        self.base2 = BaseConv(ex_channels, channels, kernel_size=3)

    def forward(self, x):
        return self.base2(self.base1(x))


class BackBone(nn.Module):
    def __init__(self):
        super(BackBone, self).__init__()
        self.conv1 = nn.Sequential(BaseConv(3, 192, kernel_size=7, stride=2))
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Sequential(BaseConv(192, 256, kernel_size=3))
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Sequential(Bottle(256),
                                   BaseConv(256, 256, kernel_size=1),
                                   BaseConv(256, 512, kernel_size=3))
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = nn.Sequential(Bottle(512),
                                   Bottle(512),
                                   Bottle(512),
                                   Bottle(512),
                                   BaseConv(512, 512, kernel_size=1),
                                   BaseConv(512, 1024, kernel_size=3))
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = nn.Sequential(Bottle(1024),
                                   Bottle(1024),
                                   BaseConv(1024, 1024, kernel_size=3),
                                   BaseConv(1024, 1024, kernel_size=3, stride=2))

        self.conv6 = nn.Sequential(BaseConv(1024, 1024, kernel_size=3),
                                   BaseConv(1024, 1024, kernel_size=3))

        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(7 * 7 * 1024, 4096), nn.LeakyReLU(),
                                nn.Linear(4096, 7 * 7 * 30), nn.Sigmoid())

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.pool4(x)

        x = self.conv5(x)
        x = self.conv6(x)

        x = self.fc(x)
        return x.reshape(-1, 7, 7, 30)


if __name__ == '__main__':
    net = BackBone()
    x = torch.randn(size=(2, 3, 448, 448))
    print(net(x).shape)
