from functools import partial

import torch
from torch import nn

# https://arxiv.org/abs/1707.01083
from Tools.utils import test_net


def channel_shuffle(x, groups):
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batch_size, -1, height, width)

    return x


class UnitB(nn.Module):
    """Figure 2(b)"""

    def __init__(self, in_channels, groups):
        super(UnitB, self).__init__()
        unit_channels = in_channels // 4

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, unit_channels, kernel_size=1, groups=groups, bias=False),
                                   nn.BatchNorm2d(unit_channels),
                                   nn.ReLU(inplace=True))
        self.shuffle = partial(channel_shuffle, groups=groups)

        self.conv2 = nn.Sequential(
            nn.Conv2d(unit_channels, unit_channels, kernel_size=3, padding=1, groups=unit_channels, bias=False),
            nn.BatchNorm2d(unit_channels))

        self.conv3 = nn.Sequential(nn.Conv2d(unit_channels, in_channels, kernel_size=1, groups=groups, bias=False),
                                   nn.BatchNorm2d(in_channels))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.conv1(x)
        y = self.shuffle(y)
        y = self.conv2(y)
        y = self.conv3(y)
        y += x
        return self.relu(y)


class UnitC(nn.Module):
    """Figure 2(c)"""

    def __init__(self, in_channels, out_channels, groups):
        super(UnitC, self).__init__()
        unit_channels = in_channels // 4

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, unit_channels, kernel_size=1, groups=groups, bias=False),
                                   nn.BatchNorm2d(unit_channels),
                                   nn.ReLU(inplace=True))
        self.shuffle = partial(channel_shuffle, groups=groups)

        self.conv2 = nn.Sequential(
            nn.Conv2d(unit_channels, unit_channels, kernel_size=3, stride=2, padding=1, groups=unit_channels,
                      bias=False),
            nn.BatchNorm2d(unit_channels))

        self.conv3 = nn.Sequential(nn.Conv2d(unit_channels, out_channels, kernel_size=1, groups=groups, bias=False),
                                   nn.BatchNorm2d(out_channels))

        self.avg = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b1 = self.avg(x)

        b2 = self.conv1(x)
        b2 = self.shuffle(b2)
        b2 = self.conv2(b2)
        b2 = self.conv3(b2)

        return torch.cat((b1, b2), dim=1)


Config = [0, 144, 200, 240, 272, 0, 0, 0, 384]


class ShuffleNet(nn.Module):
    def __init__(self, classes, groups=4):
        super(ShuffleNet, self).__init__()
        self.name = f'ShuffleNet_{groups}'
        self.channels = Config[groups]
        self.repeat = [3, 7, 3]
        in_channels = 24 * groups

        self.conv1 = nn.Sequential(nn.Conv2d(3, in_channels, kernel_size=3, stride=2, padding=1, bias=False),
                                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.conv_base = nn.Sequential()
        for i in range(3):
            out_channels = self.channels * (2 ** i)
            cat_channels = out_channels - in_channels
            self.conv_base.add_module(f'stage{i + 2}',
                                      nn.Sequential(UnitC(in_channels, cat_channels, groups),
                                                    *[UnitB(out_channels, groups)] * self.repeat[i]))
            in_channels = out_channels

        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(in_channels, classes))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv_base(x)
        return self.fc(x)

