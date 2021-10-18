import torch
from torch import nn


# https://arxiv.org/abs/1608.06993
class DenseLayer(nn.Module):
    """1x1, 3x3"""

    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        # bn_size=4
        self.conv1 = nn.Sequential(nn.BatchNorm2d(in_channels),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False))

        self.conv2 = nn.Sequential(nn.BatchNorm2d(4 * growth_rate),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return torch.cat((x, out), dim=1)


class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        super(DenseBlock, self).__init__()

        self.block = nn.Sequential()
        for i in range(num_layers):
            self.block.add_module(f'layer{i + 1}', DenseLayer(in_channels + i * growth_rate, growth_rate))

    def forward(self, x):
        return self.block(x)


class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.conv_pool = nn.Sequential(nn.BatchNorm2d(in_channels),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                                       nn.AvgPool2d(kernel_size=2))

    def forward(self, x):
        return self.conv_pool(x)


config = {'DenseNet121': (32, (6, 12, 24, 16), 64),
          'DenseNet161': (48, (6, 12, 36, 24), 96),
          'DenseNet169': (32, (6, 12, 32, 32), 64),
          'DenseNet201': (32, (6, 12, 48, 32), 64),
          'DenseNet264': (32, (6, 12, 64, 48), 64)}


class DenseNet(nn.Module):
    def __init__(self, classes, mode=121):
        super(DenseNet, self).__init__()
        self.name = f'DenseNet{mode}'

        growth_rate, block_layer, channels = config[self.name]
        self.stem = nn.Sequential(nn.Conv2d(3, channels, kernel_size=7, stride=2, padding=3, bias=False),
                                  nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
                                  nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.block_layer = block_layer
        self.back_bone = nn.Sequential()
        for i, layer in enumerate(block_layer):
            self.back_bone.add_module(f'block{i + 1}', DenseBlock(layer, channels, growth_rate))
            channels = channels + layer * growth_rate
            if i != len(block_layer) - 1:
                self.back_bone.add_module(f'trans{i + 1}', Transition(channels, channels // 2))
                channels = channels // 2

        self.fc = nn.Sequential(nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
                                nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
                                nn.Linear(channels, classes))

    def forward(self, x):
        x = self.stem(x)
        x = self.back_bone(x)
        return self.fc(x)
