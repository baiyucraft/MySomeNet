from torch import nn

# https://arxiv.org/abs/1611.05431


class BottleneckX(nn.Module):
    """基础块"""
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, base_width=64):
        super(BottleneckX, self).__init__()
        groups = 32
        width = int(out_channels * (base_width / 64.)) * groups

        self.bottle1 = nn.Sequential(nn.Conv2d(in_channels, width, kernel_size=1, stride=1, bias=False),
                                     nn.BatchNorm2d(width))
        self.bottle2 = nn.Sequential(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, groups=groups),
                                     nn.BatchNorm2d(width))
        self.bottle3 = nn.Sequential(nn.Conv2d(width, out_channels * self.expansion, kernel_size=1, bias=False),
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


# 参数 32x4d 32x8d
config = {'ResNeXt50': (4, [3, 4, 6, 3]),
          'ResNeXt101': (8, [3, 4, 23, 3])}


class ResNeXt(nn.Module):
    def __init__(self, classes, mode=50):
        super(ResNeXt, self).__init__()
        self.name = f'ResNeXt{mode}'
        self.block = BottleneckX
        self.base_width, self.layers = config[self.name]

        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.in_channels), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.conv2 = self.get_layer(self.block, channels=64, num_blocks=self.layers[0])

        self.conv3 = self.get_layer(self.block, channels=128, num_blocks=self.layers[1], stride=2)

        self.conv4 = self.get_layer(self.block, channels=256, num_blocks=self.layers[2], stride=2)

        self.conv5 = self.get_layer(self.block, channels=512, num_blocks=self.layers[3], stride=2)

        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d(1),
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
        layers = [block(self.in_channels, channels, stride, downsample, self.base_width)]
        self.in_channels = channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, channels, base_width=self.base_width))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return self.fc(x)

