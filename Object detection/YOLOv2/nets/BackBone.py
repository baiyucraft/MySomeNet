import torch
from torch import nn
from nets.block import Reorg
from utils.config import Config


class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BaseConv, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DarkNet19(nn.Module):
    def __init__(self):
        super(DarkNet19, self).__init__()
        self.conv1 = nn.Sequential(BaseConv(3, 32, kernel_size=3))
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Sequential(BaseConv(32, 64, kernel_size=3))
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Sequential(BaseConv(64, 128, kernel_size=3),
                                   BaseConv(128, 64, kernel_size=1),
                                   BaseConv(64, 128, kernel_size=3))
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = nn.Sequential(BaseConv(128, 256, kernel_size=3),
                                   BaseConv(256, 128, kernel_size=1),
                                   BaseConv(128, 256, kernel_size=3))
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = nn.Sequential(BaseConv(256, 512, kernel_size=3),
                                   BaseConv(512, 256, kernel_size=1),
                                   BaseConv(256, 512, kernel_size=3),
                                   BaseConv(512, 256, kernel_size=1),
                                   BaseConv(256, 512, kernel_size=3))

        # b1
        self.pool5 = nn.MaxPool2d(kernel_size=2)
        self.conv6 = nn.Sequential(BaseConv(512, 1024, kernel_size=3),
                                   BaseConv(1024, 512, kernel_size=1),
                                   BaseConv(512, 1024, kernel_size=3),
                                   BaseConv(1024, 512, kernel_size=1),
                                   BaseConv(512, 1024, kernel_size=3))

        self.conv7 = nn.Sequential(BaseConv(1024, 1024, kernel_size=3),
                                   BaseConv(1024, 1024, kernel_size=3))
        # b2
        self.reorg = Reorg()

        # fc
        self.conv_r = BaseConv(512 * 2 * 2 + 1024, 1024, kernel_size=3)
        out_channels = len(Config['Anchors']) * (len(Config['Classes']) + 5)
        self.fc = nn.Sequential(nn.Conv2d(1024, out_channels, kernel_size=1),
                                nn.AvgPool2d((1, 1)))

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

        b1 = self.pool5(x)
        b1 = self.conv6(b1)
        b1 = self.conv7(b1)
        b2 = self.reorg(x)

        x = torch.cat((b1, b2), dim=1)
        x = self.conv_r(x)
        return self.fc(x)


if __name__ == '__main__':
    net = DarkNet19()
    x = torch.randn(size=(2, 3, 544, 544))
    print(net(x).shape)
