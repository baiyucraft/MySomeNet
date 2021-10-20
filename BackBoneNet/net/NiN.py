from torch import nn


# https://arxiv.org/abs/1312.4400
def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding), nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(inplace=True))


class NiN(nn.Module):
    def __init__(self, classes):
        super(NiN, self).__init__()
        self.name = 'NiN'

        self.n1 = nn.Sequential(nin_block(3, 96, kernel_size=11, strides=4, padding=0),
                                nn.MaxPool2d(3, stride=2))
        self.n2 = nn.Sequential(nin_block(96, 256, kernel_size=5, strides=1, padding=2),
                                nn.MaxPool2d(3, stride=2))
        self.n3 = nn.Sequential(nin_block(256, 384, kernel_size=3, strides=1, padding=1),
                                nn.MaxPool2d(3, stride=2))
        self.n4 = nn.Sequential(nn.Dropout(0.5),
                                nin_block(384, classes, kernel_size=3, strides=1, padding=1),
                                nn.AdaptiveAvgPool2d(1),
                                nn.Flatten())

    def forward(self, x):
        x = self.n1(x)
        x = self.n2(x)
        x = self.n3(x)
        x = self.n4(x)
        return x
