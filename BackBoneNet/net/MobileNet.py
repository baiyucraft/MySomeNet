from torch import nn


# https://arxiv.org/abs/1704.04861
class BaseConv(nn.Module):
    """Figure 3 Left"""

    def __init__(self, in_channels, out_channels, stride=1):
        super(BaseConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class DWConv(nn.Module):
    """Figure 3 Right"""

    def __init__(self, in_channels, out_channels, stride=1):
        super(DWConv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels,
                      bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv2(self.conv1(x))


config = ((64, 1), (128, 2), (128, 1), (256, 2), (256, 1), (512, 2), (512, 1),
          (512, 1), (512, 1), (512, 1), (512, 1), (1024, 2), (1024, 2))


class MobileNet(nn.Module):
    def __init__(self, classes):
        super(MobileNet, self).__init__()
        self.name = 'MobileNet'
        channels = 32

        self.conv = nn.Sequential(BaseConv(3, 32, 2))

        for i, (out, stride) in enumerate(config):
            self.conv.add_module(f'Dw{i + 1}', DWConv(channels, out, stride))
            channels = out

        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                nn.Flatten(), nn.Linear(1024, classes))

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)
