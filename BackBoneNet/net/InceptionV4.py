import torch
from torch import nn

#
# V为不填充padding
from Tools.utils import test_net


class BaseConv(nn.Module):
    """基础的卷积"""

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BaseConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return self.relu(x)


class Branches(nn.Module):
    """基础branch块"""

    def __init__(self, num):
        super(Branches, self).__init__()
        self.num = num

    def forward(self, x):
        return torch.cat([getattr(self, f'branch{i + 1}')(x) for i in range(self.num)], dim=1)


class InceptionStem(nn.Module):
    """
    Figure 3
    out: 384
    """

    def __init__(self):
        super(InceptionStem, self).__init__()
        self.base1 = nn.Sequential(BaseConv(3, 32, kernel_size=3, stride=2),
                                   BaseConv(32, 32, kernel_size=3),
                                   BaseConv(32, 64, kernel_size=3, padding=1))

        self.base2_pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.base2_conv = BaseConv(64, 96, kernel_size=3, stride=2)

        self.base3_conv1 = nn.Sequential(BaseConv(160, 64, kernel_size=1),
                                         BaseConv(64, 96, kernel_size=3))
        self.base3_conv2 = nn.Sequential(BaseConv(160, 64, kernel_size=1),
                                         BaseConv(64, 64, kernel_size=(7, 1), padding=(3, 0)),
                                         BaseConv(64, 64, kernel_size=(1, 7), padding=(0, 3)),
                                         BaseConv(64, 96, kernel_size=3))

        self.base4_pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.base4_conv = BaseConv(192, 192, kernel_size=3, stride=2)

    def forward(self, x):
        x = self.base1(x)

        tmp1 = self.base2_pool(x)
        tmp2 = self.base2_conv(x)
        x = torch.cat((tmp1, tmp2), dim=1)

        tmp1 = self.base3_conv1(x)
        tmp2 = self.base3_conv2(x)
        x = torch.cat((tmp1, tmp2), dim=1)

        tmp1 = self.base4_pool(x)
        tmp2 = self.base4_conv(x)
        x = torch.cat((tmp1, tmp2), dim=1)
        return x


class InceptionA(Branches):
    """
    Figure 4
    out: 96 * 4 = 384
    """

    def __init__(self, in_channels=384):
        super(InceptionA, self).__init__(num=4)
        self.branch1 = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                                     BaseConv(in_channels, 96, kernel_size=1))

        self.branch2 = BaseConv(in_channels, 96, kernel_size=1)

        self.branch3 = nn.Sequential(BaseConv(in_channels, 64, kernel_size=1),
                                     BaseConv(64, 96, kernel_size=3, padding=1))

        self.branch4 = nn.Sequential(BaseConv(in_channels, 64, kernel_size=1),
                                     BaseConv(64, 96, kernel_size=3, padding=1),
                                     BaseConv(96, 96, kernel_size=3, padding=1))


class InceptionB(Branches):
    """
    Figure 5
    out: 128 + 384 + 256 + 256 = 1024
    """

    def __init__(self, in_channels=1024):
        super(InceptionB, self).__init__(num=4)
        self.branch1 = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                                     BaseConv(in_channels, 128, kernel_size=1))

        self.branch2 = BaseConv(in_channels, 384, kernel_size=1)

        self.branch3 = nn.Sequential(BaseConv(in_channels, 192, kernel_size=1),
                                     BaseConv(192, 224, kernel_size=(1, 7), padding=(0, 3)),
                                     BaseConv(224, 256, kernel_size=(1, 7), padding=(0, 3)))

        self.branch4 = nn.Sequential(BaseConv(in_channels, 192, kernel_size=1),
                                     BaseConv(192, 192, kernel_size=(1, 7), padding=(0, 3)),
                                     BaseConv(192, 224, kernel_size=(7, 1), padding=(3, 0)),
                                     BaseConv(224, 224, kernel_size=(1, 7), padding=(0, 3)),
                                     BaseConv(224, 256, kernel_size=(7, 1), padding=(3, 0)))


class InceptionC(nn.Module):
    """
    Figure 6
    out: 256 + 256 + 512 + 512 = 1536
    """

    def __init__(self, in_channels=1536):
        super(InceptionC, self).__init__()
        self.branch1 = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                                     BaseConv(in_channels, 256, kernel_size=1))

        self.branch2 = BaseConv(in_channels, 256, kernel_size=1)

        self.branch3 = BaseConv(in_channels, 384, kernel_size=1)
        self.branch3_conv1 = BaseConv(384, 256, kernel_size=(1, 3), padding=(0, 1))
        self.branch3_conv2 = BaseConv(384, 256, kernel_size=(3, 1), padding=(1, 0))

        self.branch4 = nn.Sequential(BaseConv(in_channels, 384, kernel_size=1),
                                     BaseConv(384, 448, kernel_size=(1, 3), padding=(0, 1)),
                                     BaseConv(448, 512, kernel_size=(3, 1), padding=(1, 0)))
        self.branch4_conv1 = BaseConv(512, 256, kernel_size=(3, 1), padding=(1, 0))
        self.branch4_conv2 = BaseConv(512, 256, kernel_size=(1, 3), padding=(0, 1))

    def forward(self, x):
        b1 = self.branch1(x)

        b2 = self.branch2(x)

        b3 = self.branch3(x)
        tmp1 = self.branch3_conv1(b3)
        tmp2 = self.branch3_conv2(b3)
        b3 = torch.cat((tmp1, tmp2), dim=1)

        b4 = self.branch4(x)
        tmp1 = self.branch4_conv1(b4)
        tmp2 = self.branch4_conv2(b4)
        b4 = torch.cat((tmp1, tmp2), dim=1)
        return torch.cat((b1, b2, b3, b4), dim=1)


class ReductionA(Branches):
    """
    Figure 7
    out: in_channels + in_channels + m = 768 + m
    """

    def __init__(self, in_channels, k, l, m, n=384):
        super(ReductionA, self).__init__(num=3)
        self.branch1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.branch2 = BaseConv(in_channels, n, kernel_size=3, stride=2)

        self.branch3 = nn.Sequential(BaseConv(in_channels, k, kernel_size=1),
                                     BaseConv(k, l, kernel_size=3, padding=1),
                                     BaseConv(l, m, kernel_size=3, stride=2))


class ReductionB(Branches):
    """
    Figure 8
    out: in_channels(1024) + 192 + 320 = 1536
    """

    def __init__(self, in_channels):
        super(ReductionB, self).__init__(num=3)
        self.branch1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.branch2 = nn.Sequential(BaseConv(in_channels, 192, kernel_size=1),
                                     BaseConv(192, 192, kernel_size=3, stride=2))

        self.branch3 = nn.Sequential(BaseConv(in_channels, 256, kernel_size=1),
                                     BaseConv(256, 256, kernel_size=(1, 7), padding=(0, 3)),
                                     BaseConv(256, 320, kernel_size=(7, 1), padding=(3, 0)),
                                     BaseConv(320, 320, kernel_size=3, stride=2))


# --------------------
# Inception 结合 ResNet
# --------------------

class InceptionStemRes(nn.Module):
    """
    Figure 14
    out: 256
    """

    def __init__(self):
        super(InceptionStemRes, self).__init__()
        self.base1 = nn.Sequential(BaseConv(3, 32, kernel_size=3, stride=2),
                                   BaseConv(32, 32, kernel_size=3),
                                   BaseConv(32, 64, kernel_size=3, padding=1),
                                   nn.MaxPool2d(kernel_size=3, stride=2))

        self.base2 = nn.Sequential(BaseConv(64, 80, kernel_size=1),
                                   BaseConv(80, 192, kernel_size=3),
                                   BaseConv(192, 256, kernel_size=3, stride=2))

    def forward(self, x):
        x = self.base1(x)
        x = self.base2(x)
        return x


class InceptionResA(nn.Module):
    """
    Figure 10 or 16
    out: channels(256 or 384)
    """

    def __init__(self, channels, mode):
        super(InceptionResA, self).__init__()
        b3_1, b3_2, b3_3 = (32, 32, 32) if mode == 'V1' else (32, 48, 64)

        self.branch1 = BaseConv(channels, 32, kernel_size=1)

        self.branch2 = nn.Sequential(BaseConv(channels, 32, kernel_size=1),
                                     BaseConv(32, 32, kernel_size=3, padding=1))

        self.branch3 = nn.Sequential(BaseConv(channels, b3_1, kernel_size=1),
                                     BaseConv(b3_1, b3_2, kernel_size=3, padding=1),
                                     BaseConv(b3_2, b3_3, kernel_size=3, padding=1))

        self.conv_linear = nn.Conv2d(64 + b3_3, channels, kernel_size=1, bias=True)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b = torch.cat((b1, b2, b3), dim=1)
        y = self.conv_linear(b)
        y += x
        return self.relu(y)


class InceptionResB(nn.Module):
    """
    Figure 11 or 17
    out: channels(896 or 1152) -- no 1154
    """

    def __init__(self, channels, mode):
        super(InceptionResB, self).__init__()
        b2_1, b2_2, b2_3 = (128, 128, 128) if mode == 'V1' else (128, 160, 192)

        self.branch1 = BaseConv(channels, 128, kernel_size=1)

        self.branch2 = nn.Sequential(BaseConv(channels, b2_1, kernel_size=1),
                                     BaseConv(b2_1, b2_2, kernel_size=(1, 7), padding=(0, 3)),
                                     BaseConv(b2_2, b2_3, kernel_size=(7, 1), padding=(3, 0)))

        self.conv_linear = nn.Conv2d(128 + b2_3, channels, kernel_size=1, bias=True)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b = torch.cat((b1, b2), dim=1)
        y = self.conv_linear(b)
        y += x
        return self.relu(y)


class InceptionResC(nn.Module):
    """
    Figure 13 or 19
    out: channels(1792 or 2144) -- no2048
    """

    def __init__(self, channels, mode):
        super(InceptionResC, self).__init__()
        b2_1, b2_2, b2_3 = (192, 192, 192) if mode == 'V1' else (192, 224, 256)

        self.branch1 = BaseConv(channels, 192, kernel_size=1)

        self.branch2 = nn.Sequential(BaseConv(channels, b2_1, kernel_size=1),
                                     BaseConv(b2_1, b2_2, kernel_size=(1, 3), padding=(0, 1)),
                                     BaseConv(b2_2, b2_3, kernel_size=(3, 1), padding=(1, 0)))

        self.conv_linear = nn.Conv2d(192 + b2_3, channels, kernel_size=1, bias=True)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b = torch.cat((b1, b2), dim=1)
        y = self.conv_linear(b)
        y += x
        return self.relu(y)


class ReductionResB(Branches):
    """
    Figure 12 or 18
    out: 896(1154) + 384 + 256(288) + 256(320) = 1792(1248)
    """

    def __init__(self, in_channels, mode):
        super(ReductionResB, self).__init__(num=4)
        b3, b4_1, b4_2, b4_3 = (256, 256, 256, 256) if mode == 'V1' else (288, 256, 288, 320)

        self.branch1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.branch2 = nn.Sequential(BaseConv(in_channels, 256, kernel_size=1),
                                     BaseConv(256, 384, kernel_size=3, stride=2))

        self.branch3 = nn.Sequential(BaseConv(in_channels, 256, kernel_size=1),
                                     BaseConv(256, b3, kernel_size=3, stride=2))

        self.branch4 = nn.Sequential(BaseConv(in_channels, b4_1, kernel_size=1),
                                     BaseConv(b4_1, b4_2, kernel_size=3, padding=1),
                                     BaseConv(b4_2, b4_3, kernel_size=3, stride=2))


# 论文中的 1154 和 2048 错误，改为 1152 和 2144
config = {'V4': (192, 224, 256),
          'ResV1': (192, 192, 256, [256, 896, 1792]),
          'ResV2': (256, 256, 384, [384, 1152, 2144])}


class InceptionV4(nn.Module):
    def __init__(self, classes):
        super(InceptionV4, self).__init__()
        self.name = 'InceptionV4'
        k, l, m = config['V4']

        self.stem = InceptionStem()

        self.inception1 = nn.Sequential(*[InceptionA()] * 4)
        self.reduction1 = ReductionA(384, k, l, m)

        self.inception2 = nn.Sequential(*[InceptionB()] * 7)
        self.reduction2 = ReductionB(1024)

        self.inception3 = nn.Sequential(*[InceptionC()] * 3)

        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Dropout(0.8),
                                nn.Linear(1536, classes))

    def forward(self, x):
        x = self.stem(x)
        x = self.inception1(x)
        x = self.reduction1(x)
        x = self.inception2(x)
        x = self.reduction2(x)
        x = self.inception3(x)
        return self.fc(x)


class InceptionRes(nn.Module):
    def __init__(self, classes, mode='V1'):
        super(InceptionRes, self).__init__()
        self.name = f'InceptionRes{mode}'
        k, l, m, layer = config[f'Res{mode}']

        self.stem = InceptionStemRes() if mode == 'V1' else InceptionStem()

        self.inception1 = nn.Sequential(*[InceptionResA(layer[0], mode)] * 5)
        self.reduction1 = ReductionA(layer[0], k, l, m)

        self.inception2 = nn.Sequential(*[InceptionResB(layer[1], mode)] * 10)
        self.reduction2 = ReductionResB(layer[1], mode)

        self.inception3 = nn.Sequential(*[InceptionResC(layer[2], mode)] * 5)

        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Dropout(0.8),
                                nn.Linear(layer[2], classes))

    def forward(self, x):
        x = self.stem(x)
        x = self.inception1(x)
        x = self.reduction1(x)
        x = self.inception2(x)
        x = self.reduction2(x)
        x = self.inception3(x)
        return self.fc(x)
