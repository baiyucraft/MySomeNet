import torch
from torch import nn


# https://arxiv.org/abs/1502.03167
# https://arxiv.org/abs/1512.00567
class BaseConv(nn.Module):
    """基础的卷积"""

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BaseConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class InceptionA(nn.Module):
    """Figure5：
    out:64+64+96+pool_features = 224+pool_features"""

    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1 = BaseConv(in_channels, 64, kernel_size=1)

        self.branch2 = nn.Sequential(BaseConv(in_channels, 48, kernel_size=1),
                                     BaseConv(48, 64, kernel_size=5, padding=2))

        self.branch3 = nn.Sequential(BaseConv(in_channels, 64, kernel_size=1),
                                     BaseConv(64, 96, kernel_size=3, padding=1),
                                     BaseConv(96, 96, kernel_size=3, padding=1))

        self.branch4 = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                                     BaseConv(in_channels, pool_features, kernel_size=1))

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        return torch.cat((branch1, branch2, branch3, branch4), dim=1)


class InceptionB(nn.Module):
    """out:384+96+in_channels = 480+in_channels"""

    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch1 = BaseConv(in_channels, 384, kernel_size=3, stride=2)

        self.branch2 = nn.Sequential(BaseConv(in_channels, 64, kernel_size=1),
                                     BaseConv(64, 96, kernel_size=3, padding=1),
                                     BaseConv(96, 96, kernel_size=3, stride=2))

        self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        return torch.cat((branch1, branch2, branch3), dim=1)


class InceptionC(nn.Module):
    """Figure6:
    out:192*4 = 768"""

    def __init__(self, in_channels, channels):
        super(InceptionC, self).__init__()
        self.branch1 = BaseConv(in_channels, 192, kernel_size=1)

        self.branch2 = nn.Sequential(BaseConv(in_channels, channels, kernel_size=1),
                                     BaseConv(channels, channels, kernel_size=(1, 7), padding=(0, 3)),
                                     BaseConv(channels, 192, kernel_size=(7, 1), padding=(3, 0)))

        self.branch3 = nn.Sequential(BaseConv(in_channels, channels, kernel_size=1),
                                     BaseConv(channels, channels, kernel_size=(7, 1), padding=(3, 0)),
                                     BaseConv(channels, channels, kernel_size=(1, 7), padding=(0, 3)),
                                     BaseConv(channels, channels, kernel_size=(7, 1), padding=(3, 0)),
                                     BaseConv(channels, 192, kernel_size=(1, 7), padding=(0, 3)))

        self.branch4 = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                                     BaseConv(in_channels, 192, kernel_size=1))

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        return torch.cat((branch1, branch2, branch3, branch4), dim=1)


class InceptionD(nn.Module):
    """out:320+192+768"""

    def __init__(self, in_channels):
        super(InceptionD, self).__init__()
        self.branch1 = nn.Sequential(BaseConv(in_channels, 192, kernel_size=1),
                                     BaseConv(192, 320, kernel_size=3, stride=2))

        self.branch2 = nn.Sequential(BaseConv(in_channels, 192, kernel_size=1),
                                     BaseConv(192, 192, kernel_size=(1, 7), padding=(0, 3)),
                                     BaseConv(192, 192, kernel_size=(7, 1), padding=(3, 0)),
                                     BaseConv(192, 192, kernel_size=3, stride=2))

        self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        return torch.cat((branch1, branch2, branch3), dim=1)


class InceptionE(nn.Module):
    """Figure7:
    out:320+384+384+384+384+192 = 2048"""

    def __init__(self, in_channels):
        super(InceptionE, self).__init__()
        self.branch1 = BaseConv(in_channels, 320, kernel_size=1)

        self.branch2 = BaseConv(in_channels, 384, kernel_size=1)
        self.branch2_a = BaseConv(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch2_b = BaseConv(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3 = nn.Sequential(BaseConv(in_channels, 448, kernel_size=1),
                                     BaseConv(448, 384, kernel_size=3, padding=1))
        self.branch3_a = BaseConv(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3_b = BaseConv(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch4 = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                                     BaseConv(in_channels, 192, kernel_size=1))

    def forward(self, x):
        branch1 = self.branch1(x)

        branch2 = self.branch2(x)
        branch2_a = self.branch2_a(branch2)
        branch2_b = self.branch2_b(branch2)
        branch2 = torch.cat((branch2_a, branch2_b), dim=1)

        branch3 = self.branch3(x)
        branch3_a = self.branch3_a(branch3)
        branch3_b = self.branch3_b(branch3)
        branch3 = torch.cat((branch3_a, branch3_b), dim=1)

        branch4 = self.branch4(x)
        return torch.cat((branch1, branch2, branch3, branch4), dim=1)


class InceptionV23(nn.Module):
    def __init__(self, classes):
        super(InceptionV23, self).__init__()
        self.name = 'InceptionV23'

        in_channels = 3
        self.conv1 = nn.Sequential(BaseConv(in_channels, 32, kernel_size=3, stride=2),
                                   BaseConv(32, 32, kernel_size=3, stride=1),
                                   BaseConv(32, 64, kernel_size=3, stride=1, padding=1),
                                   nn.MaxPool2d(kernel_size=3, stride=2))

        self.conv2 = nn.Sequential(BaseConv(64, 80, kernel_size=1),
                                   BaseConv(80, 192, kernel_size=3),
                                   nn.MaxPool2d(kernel_size=3, stride=2))

        self.inception3 = nn.Sequential(InceptionA(192, pool_features=32),
                                        InceptionA(256, pool_features=64),
                                        InceptionA(288, pool_features=64))

        self.inception4 = nn.Sequential(InceptionB(288),
                                        InceptionC(768, channels=128),
                                        InceptionC(768, channels=160),
                                        InceptionC(768, channels=160),
                                        InceptionC(768, channels=192))

        self.inception5 = nn.Sequential(InceptionD(768),
                                        InceptionE(1280),
                                        InceptionE(2048))

        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                nn.Flatten(), nn.Dropout(),
                                nn.Linear(2048, classes))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.inception3(x)
        x = self.inception4(x)
        x = self.inception5(x)
        return self.fc(x)
