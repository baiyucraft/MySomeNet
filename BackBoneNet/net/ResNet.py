import torch
from torch import nn


# https://arxiv.org/abs/1512.03385

class BasicBlock(nn.Module):
    """18 34的基础残差块"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.basic1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                                    nn.BatchNorm2d(out_channels))
        self.basic2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        y = self.relu(self.basic1(x))
        y = self.basic2(y)
        if self.downsample is not None:
            x = self.downsample(x)
        y += x
        return self.relu(y)


class Bottleneck(nn.Module):
    """50 101 152的基础残差块"""
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.bottle1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                                     nn.BatchNorm2d(out_channels))
        self.bottle2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                                     nn.BatchNorm2d(out_channels))
        self.bottle3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion))
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        y = self.relu(self.bottle1(x))
        y = self.relu(self.bottle2(y))
        y = self.bottle3(y)

        if self.downsample is not None:
            x = self.downsample(x)
        y += x
        return self.relu(y)


# 参数
resnet_18 = (BasicBlock, [2, 2, 2, 2])
resnet_34 = (BasicBlock, [3, 4, 6, 3])
resnet_50 = (Bottleneck, [3, 4, 6, 3])
resnet_101 = (Bottleneck, [3, 4, 23, 3])
resnet_152 = (Bottleneck, [3, 8, 36, 3])


class ResNet(nn.Module):
    def __init__(self, classes, mode=50):
        super(ResNet, self).__init__()
        self.name = f'ResNet{mode}'
        self.block, self.layers = globals()[f'resnet_{mode}']

        in_channels = 3
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.in_channels), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.conv2 = self.get_layer(self.block, channels=64, num_blocks=self.layers[0])

        self.conv3 = self.get_layer(self.block, channels=128, num_blocks=self.layers[1], stride=2)

        self.conv4 = self.get_layer(self.block, channels=256, num_blocks=self.layers[2], stride=2)

        self.conv5 = self.get_layer(self.block, channels=512, num_blocks=self.layers[3], stride=2)

        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                nn.Flatten(),
                                nn.Linear(512 * self.block.expansion, classes))

    def get_layer(self, block, channels, num_blocks, stride=1):
        downsample = None
        # stride 不为 1的话升维
        if stride != 1 or self.in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels * block.expansion),
            )
        layers = [block(self.in_channels, channels, stride, downsample)]
        self.in_channels = channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return self.fc(x)
