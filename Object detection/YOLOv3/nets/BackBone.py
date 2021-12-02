import torch
from torch import nn
from nets.block import Reorg
from utils.config import Config

cfg = [[1, 64, 128],
       [3, 128, 256],
       [8, 256, 512],
       [8, 512, 1024],
       [4, 1024, 1024]]


class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BaseConv, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DarkNetBlock(nn.Module):
    def __init__(self, channels):
        super(DarkNetBlock, self).__init__()
        ex_channels = channels // 2
        self.conv1 = BaseConv(channels, ex_channels, kernel_size=1)
        self.conv2 = BaseConv(ex_channels, channels, kernel_size=3)

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        return y + x


def make_layers(r, channels):
    layers = []
    for _ in range(r):
        layers.append(DarkNetBlock(channels))
    return nn.Sequential(*layers)


class CTrans(nn.Module):
    def __init__(self, anchors):
        super(CTrans, self).__init__()
        self.size = Config['Size'][0]
        self.num_classes = len(Config['Classes'])
        self.anchors = torch.Tensor(anchors) * Config['Size'][0]
        self.num_anchors = len(self.anchors)

    def forward(self, c):
        device = c.device
        batch_size, _, h, w = c.shape
        grid_size = h
        bbox_attrs = 5 + self.num_classes
        num_anchors = len(self.anchors)

        c = c.reshape(batch_size, num_anchors, bbox_attrs, grid_size, grid_size).permute(0, 1, 3, 4, 2)

        if self.training:
            return c
        else:
            prediction = c.sigmoid()

            grid, anchor_grid = self._make_grid(grid_size, device)

            prediction[..., 0:2] = (prediction[..., 0:2] * 2 - 0.5 + grid) / grid_size
            prediction[..., 2:4] = (prediction[..., 2:4] * 2) ** 2 * anchor_grid / self.size

            prediction = prediction.reshape(batch_size, -1, bbox_attrs)
            return prediction

    def _make_grid(self, grid_size, device):
        grid_len = torch.arange(grid_size, device=device)
        c_i, c_j = torch.meshgrid(grid_len, grid_len)
        grid = torch.stack((c_i, c_j), 2).expand((1, self.num_anchors, grid_size, grid_size, 2)).float()

        anchor_grid = self.anchors.clone().to(device).reshape((1, self.num_anchors, 1, 1, 2))
        anchor_grid = anchor_grid.expand((1, self.num_anchors, grid_size, grid_size, 2))
        return grid, anchor_grid


def make_conv_set(in_channels, out_channels):
    layers = []
    ex_channels = out_channels * 2
    layers.append(BaseConv(in_channels, out_channels, kernel_size=1))
    layers.append(BaseConv(out_channels, ex_channels, kernel_size=3))
    layers.append(BaseConv(ex_channels, out_channels, kernel_size=1))
    layers.append(BaseConv(out_channels, ex_channels, kernel_size=3))
    layers.append(BaseConv(ex_channels, out_channels, kernel_size=1))
    return nn.Sequential(*layers)


class DarkNet53(nn.Module):
    def __init__(self):
        super(DarkNet53, self).__init__()
        self.num_anchors = len(Config['Anchors']) // 3
        self.num_classes = len(Config['Classes'])
        # darknet53
        self.conv1 = nn.Sequential(BaseConv(3, 32, kernel_size=3))
        self.pool1 = nn.Sequential(BaseConv(32, 64, kernel_size=3, stride=2))

        n, in_channels, out_channels = cfg[0]
        self.conv2 = make_layers(n, in_channels)
        self.pool2 = nn.Sequential(BaseConv(in_channels, out_channels, kernel_size=3, stride=2))

        n, in_channels, out_channels = cfg[1]
        self.conv3 = make_layers(n, in_channels)
        self.pool3 = nn.Sequential(BaseConv(in_channels, out_channels, kernel_size=3, stride=2))

        n, in_channels, out_channels = cfg[2]
        self.conv4 = make_layers(n, in_channels)
        self.pool4 = nn.Sequential(BaseConv(in_channels, out_channels, kernel_size=3, stride=2))

        n, in_channels, out_channels = cfg[3]
        self.conv5 = make_layers(n, in_channels)
        self.pool5 = nn.Sequential(BaseConv(in_channels, out_channels, kernel_size=3, stride=2))

        n, in_channels, out_channels = cfg[4]
        self.conv6 = make_layers(n, in_channels)

        # c3
        self.conv_set1 = make_conv_set(out_channels, 512)
        self.c3_out = nn.Sequential(BaseConv(512, 1024, kernel_size=1),
                                    nn.Conv2d(1024, self.num_anchors * (5 + self.num_classes), kernel_size=1))

        # c2
        self.yolo_base1 = nn.Sequential(BaseConv(512, 256, kernel_size=1),
                                        nn.Upsample(scale_factor=2, mode="nearest"))
        # cat
        self.conv_set2 = make_conv_set(768, 256)
        self.c2_out = nn.Sequential(BaseConv(256, 512, kernel_size=3),
                                    nn.Conv2d(512, self.num_anchors * (5 + self.num_classes), kernel_size=1))

        # c1
        self.yolo_base2 = nn.Sequential(BaseConv(256, 128, kernel_size=3),
                                        nn.Upsample(scale_factor=2, mode="nearest"))
        # cat
        self.conv_set3 = make_conv_set(384, 128)
        self.c1_out = nn.Sequential(BaseConv(128, 256, kernel_size=3),
                                    nn.Conv2d(256, self.num_anchors * (5 + self.num_classes), kernel_size=1))

        self.c1_pred = CTrans(Config['Anchors'][:3])
        self.c2_pred = CTrans(Config['Anchors'][3:6])
        self.c3_pred = CTrans(Config['Anchors'][6:])

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.pool3(x)

        c1 = self.conv4(x)
        x = self.pool4(c1)

        c2 = self.conv5(x)
        x = self.pool5(c2)

        x = self.conv6(x)

        # c1
        x = self.conv_set1(x)
        c3 = self.c3_out(x)

        # c2
        x = self.yolo_base1(x)
        x = torch.cat((c2, x), dim=1)
        x = self.conv_set2(x)
        c2 = self.c2_out(x)

        # c3
        x = self.yolo_base2(x)
        x = torch.cat((c1, x), dim=1)
        x = self.conv_set3(x)
        c1 = self.c1_out(x)

        # print(c1.shape, c2.shape, c3.shape)
        # /8
        c1 = self.c1_pred(c1)
        # /16
        c2 = self.c2_pred(c2)
        # /32
        c3 = self.c3_pred(c3)

        if self.training:
            return c1, c2, c3
        else:
            return torch.cat((c1, c2, c3), dim=1)


if __name__ == '__main__':
    net = DarkNet53()
    x = torch.randn(size=(2, 3, 256, 256))
    print(net)
    print(net(x))
