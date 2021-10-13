import torch
from torch import nn


class Inception(nn.Module):
    def __init__(self, in_channels, c1, c13, c15, c31):
        super(Inception, self).__init__()
        # branch1，1x1 conv
        self.branch1 = nn.Sequential(nn.Conv2d(in_channels, c1, kernel_size=1), nn.ReLU())
        # branch2，1x1 conv + 3x3 conv
        self.branch2 = nn.Sequential(nn.Conv2d(in_channels, c13[0], kernel_size=1), nn.ReLU(),
                                     nn.Conv2d(c13[0], c13[1], kernel_size=3, padding=1), nn.ReLU())
        # branch3，1x1 conv + 5x5 conv
        self.branch3 = nn.Sequential(nn.Conv2d(in_channels, c15[0], kernel_size=1), nn.ReLU(),
                                     nn.Conv2d(c15[0], c15[1], kernel_size=5, padding=2), nn.ReLU())
        # branch5，3x3 max_pool + 1x1 conv
        self.branch4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                                     nn.Conv2d(in_channels, c31, kernel_size=1), nn.ReLU())

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        # 在通道维度上连结输出
        return torch.cat((branch1, branch2, branch3, branch4), dim=1)


class GoogLeNet(nn.Module):
    def __init__(self, classes):
        super(GoogLeNet, self).__init__()
        self.name = 'GoogLeNet'

        in_channels = 3
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3), nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1), nn.ReLU(),
                                   nn.Conv2d(64, 192, kernel_size=3, padding=1), nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.inception3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                                        Inception(256, 128, (128, 192), (32, 96), 64),
                                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.inception4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                                        Inception(512, 160, (112, 224), (24, 64), 64),
                                        Inception(512, 128, (128, 256), (24, 64), 64),
                                        Inception(512, 112, (144, 288), (32, 64), 64),
                                        Inception(528, 256, (160, 320), (32, 128), 128),
                                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.inception5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                                        Inception(832, 384, (192, 384), (48, 128), 128),
                                        nn.AdaptiveAvgPool2d((1, 1)))

        self.fc = nn.Sequential(nn.Flatten(), nn.Dropout(p=0.4),
                                nn.Linear(1024, classes))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.inception3(x)
        x = self.inception4(x)
        x = self.inception5(x)
        return self.fc(x)
