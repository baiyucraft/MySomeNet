from torch import nn
from torch.nn import functional as F
from torchvision.ops.misc import FrozenBatchNorm2d


class Bottleneck(nn.Module):
    """50 101 152的基础残差块"""
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.bottle1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                                     FrozenBatchNorm2d(out_channels))
        self.bottle2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                                     FrozenBatchNorm2d(out_channels))
        self.bottle3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, bias=False),
            FrozenBatchNorm2d(out_channels * self.expansion))
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


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.block, self.layers = (Bottleneck, [3, 4, 6, 3])

        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False),
            FrozenBatchNorm2d(self.in_channels), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.conv2 = self.get_layer(self.block, channels=64, num_blocks=self.layers[0])

        self.conv3 = self.get_layer(self.block, channels=128, num_blocks=self.layers[1], stride=2)

        self.conv4 = self.get_layer(self.block, channels=256, num_blocks=self.layers[2], stride=2)

        self.conv5 = self.get_layer(self.block, channels=512, num_blocks=self.layers[3], stride=2)

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
        x1 = self.conv2(x)
        x2 = self.conv3(x1)
        x3 = self.conv4(x2)
        x4 = self.conv5(x3)
        return [x1, x2, x3, x4]


class FeaturePyramidNetwork(nn.Module):
    """增强特征图"""

    def __init__(self, in_channels_list, out_channels):
        super(FeaturePyramidNetwork, self).__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        for in_channels in in_channels_list:
            inner_block_module = nn.Conv2d(in_channels, out_channels, 1)
            layer_block_module = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)

    def forward(self, x):
        # 特征图
        c1 = self.inner_blocks[0](x[0])
        c2 = self.inner_blocks[1](x[1])
        c3 = self.inner_blocks[2](x[2])
        c4 = self.inner_blocks[3](x[3])

        c3 = F.interpolate(c4, size=c3.shape[-2:], mode='nearest') + c3
        c2 = F.interpolate(c3, size=c2.shape[-2:], mode='nearest') + c2
        c1 = F.interpolate(c2, size=c1.shape[-2:], mode='nearest') + c1

        c1 = self.layer_blocks[0](c1)
        c2 = self.layer_blocks[1](c2)
        c3 = self.layer_blocks[2](c3)
        c4 = self.layer_blocks[3](c4)
        pool = F.max_pool2d(c4, 1, 2, 0)

        out = [c1, c2, c3, c4, pool]
        # out = OrderedDict([(k, v) for k, v in zip(names, results)])

        return out


class BackBone(nn.Module):
    """主干网络"""
    def __init__(self):
        super(BackBone, self).__init__()
        channels_list = [256, 512, 1024, 2048]
        out_channels = 256

        self.body = ResNet()
        self.fpn = FeaturePyramidNetwork(channels_list, out_channels)

    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)
        # [c1,c2,c3,c4,pool]
        return x

