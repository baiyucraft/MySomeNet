import torch
from torch import nn


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


class Block(nn.Module):
    def __init__(self, in_p, out_p, mode='2d'):
        super(Block, self).__init__()
        if mode == '3d':
            conv = nn.Conv3d
            bn = nn.BatchNorm3d
        else:
            conv = nn.Conv2d
            bn = nn.BatchNorm2d

        hid_p = out_p // 4
        self.bottle1 = nn.Sequential(conv(in_p, hid_p, kernel_size=1, stride=1, groups=hid_p, bias=False),
                                     bn(hid_p), nn.ReLU(inplace=True))
        # self.bottle2 = nn.Sequential(conv(hid_p, hid_p, kernel_size=3, stride=1, padding=1, groups=hid_p, bias=False),
        #                              bn(hid_p), nn.ReLU(inplace=True))
        # self.bottle3 = nn.Sequential(conv(hid_p, out_p, kernel_size=1, stride=1, bias=False),
        #                              bn(out_p))
        self.com = nn.Sequential(conv(hid_p, out_p, kernel_size=3, stride=1, padding=1, bias=False),
                                 bn(out_p))
        self.r = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.bottle1(x)
        # y = self.bottle2(y)
        # y = self.bottle3(y)
        y = self.com(y)
        y += x
        return self.r(y)


class Pool(nn.Module):
    def __init__(self, in_p, out_p, mode='2d'):
        super(Pool, self).__init__()
        if mode == '3d':
            conv = nn.Conv3d
            bn = nn.BatchNorm3d
        else:
            conv = nn.Conv2d
            bn = nn.BatchNorm2d

        # self.conv = nn.Sequential(conv(in_p, out_p, kernel_size=3, stride=2, padding=1, groups=in_p, bias=False),
        #                           bn(out_p), nn.ReLU(inplace=True))
        # self.po = nn.Sequential(conv(out_p, out_p, kernel_size=1, stride=1, bias=False),
        #                         bn(out_p), nn.ReLU(inplace=True))
        self.com = nn.Sequential(conv(in_p, out_p, kernel_size=3, stride=2, padding=1),
                                 bn(out_p), nn.ReLU(inplace=True))

    def forward(self, x):
        # y = self.conv(x)
        # y = self.po(y)
        y = self.com(x)
        return y


def test_net2():
    x = torch.Tensor(1, 1, 125, 32, 32)

    net = nn.Sequential(Block(1, 4, mode='3d'), Pool(4, 8, mode='3d'),
                        Block(8, 8, mode='3d'), Pool(8, 16, mode='3d'),
                        Block(16, 16, mode='3d'), Pool(16, 32, mode='3d'),
                        nn.Flatten(start_dim=1, end_dim=2),
                        Block(512, 512), Pool(512, 1024),
                        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(1024, 4))

    print(x.shape)
    for layer in net:
        x = layer(x)
        print(x.shape)

    print(get_parameter_number(net))
    torch.save(net.state_dict(), 'model_data/tmp1.pt')


def test_net1():
    x = torch.Tensor(1, 6, 224, 224)

    conv1 = nn.Sequential(nn.Conv2d(6, 32, kernel_size=7, stride=2, padding=3),
                          nn.BatchNorm2d(32), nn.ReLU(inplace=True))
    net = nn.Sequential(conv1, Pool(32, 64),  # 64*56*56
                        Block(64, 64), Block(64, 64), Pool(64, 128),  # 128*28*28
                        Block(128, 128), Block(128, 128), Pool(128, 256),  # 256*14*14
                        Block(256, 256), Block(256, 256), Pool(256, 512),  # 512*7*7
                        Block(512, 512), Block(512, 512),  # 512*7*7
                        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, 8))

    print(x.shape)
    for layer in net:
        x = layer(x)
        print(x.shape)

    print(get_parameter_number(net))
    torch.save(net.state_dict(), 'model_data/tmp2.pt')


if __name__ == '__main__':
    test_net1()
    # test_net2()
