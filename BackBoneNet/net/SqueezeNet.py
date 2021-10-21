import torch
from torch import nn
from functools import partial


# https://arxiv.org/abs/1602.07360
class FireBlock(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1_channels, expand2_channels):
        super(FireBlock, self).__init__()
        self.squeeze = nn.Sequential(nn.Conv2d(in_channels, squeeze_channels, kernel_size=1),
                                     nn.ReLU(inplace=True))

        self.expand1 = nn.Sequential(nn.Conv2d(squeeze_channels, expand1_channels, kernel_size=1),
                                     nn.ReLU(inplace=True))

        self.expand3 = nn.Sequential(nn.Conv2d(squeeze_channels, expand2_channels, kernel_size=3, padding=1),
                                     nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.squeeze(x)
        e1 = self.expand1(x)
        e2 = self.expand3(x)
        return torch.cat((e1, e2), dim=1)


config = ((96, 16, 64, 64),
          (128, 16, 64, 64),
          (128, 32, 128, 128),
          (0, 0, 0, 0),
          (256, 32, 128, 128),
          (256, 48, 192, 192),
          (384, 48, 192, 192),
          (384, 64, 256, 256),
          (0, 0, 0, 0),
          (512, 64, 256, 256))


class SqueezeNet(nn.Module):
    def __init__(self, classes):
        super(SqueezeNet, self).__init__()
        self.name = 'SqueezeNet'
        pool = partial(nn.MaxPool2d, kernel_size=3, stride=2, ceil_mode=True)

        self.conv1 = nn.Sequential(nn.Conv2d(3, 96, kernel_size=7, stride=2),
                                   nn.ReLU(inplace=True),
                                   pool())

        self.base = nn.Sequential()
        for i, (in_c, squeeze_c, e1_c, e3_c) in enumerate(config):
            if in_c != 0:
                self.base.add_module(f'fire{i + 2}', FireBlock(in_c, squeeze_c, e1_c, e3_c))
            else:
                self.base.add_module(f'pool{i + 2}', pool())

        self.fc = nn.Sequential(nn.Dropout(p=0.5),
                                nn.Conv2d(512, classes, kernel_size=1),
                                nn.ReLU(inplace=True),
                                nn.AdaptiveAvgPool2d(1),
                                nn.Flatten())

    def forward(self, x):
        x = self.conv1(x)
        x = self.base(x)
        return self.fc(x)
