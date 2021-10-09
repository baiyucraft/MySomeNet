import torch
from torch import nn

"""
最后一个池化不降高宽
conv6、conv7。
"""

conv_arch = [(2, 64), (2, 128), (3, 256), (3, 512), (3, 512)]


def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
    return nn.Sequential(*layers)


def vgg(i):
    layers = nn.Sequential()
    in_channels = i
    # 卷积层部分
    for i, (num_convs, out_channels) in enumerate(conv_arch):
        layers.add_module(f'conv{i + 1}_3', vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels
        # 修改最后一个池化
    layers[-1][-1] = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    # conv6 conv7
    conv6 = nn.Sequential(nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6), nn.ReLU(inplace=True))
    conv7 = nn.Sequential(nn.Conv2d(1024, 1024, kernel_size=1), nn.ReLU(inplace=True))
    layers.add_module('conv6', conv6)
    layers.add_module('conv7', conv7)
    return layers


def get_layer(net, X):
    """得到每层输出"""
    # print(net)
    for layers in net:
        for layer in layers:
            X = layer(X)
            print(layer.__class__.__name__, 'output shape: \t', X.shape)
        print(' ')


if __name__ == '__main__':
    X = torch.randn(size=(1, 3, 300, 300))
    get_layer(vgg(3), X)
