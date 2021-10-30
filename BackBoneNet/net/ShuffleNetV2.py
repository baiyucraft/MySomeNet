import torch
from torch import nn


# https://arxiv.org/abs/1807.11164

def channel_shuffle(x, groups):
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batch_size, -1, height, width)

    return x


class Unit(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Unit, self).__init__()
        self.stride = stride
        branch_channels = out_channels // 2

        if stride > 1:
            self.branch1 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels,
                          bias=False),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, branch_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels if stride > 1 else branch_channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3, stride=stride, padding=1, groups=branch_channels,
                      bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
        else:
            x1, x2 = x, x

        b1 = self.branch1(x1)
        b2 = self.branch2(x2)
        y = torch.cat((b1, b2), dim=1)

        return channel_shuffle(y, 2)


Config = {
    'ShuffleNetV2_x0_5': (24, 48, 96, 192, 1024),
    'ShuffleNetV2_x1_0': (24, 116, 232, 464, 1024),
    'ShuffleNetV2_x1_5': (24, 176, 352, 704, 1024),
    'ShuffleNetV2_x2_0': (24, 244, 488, 976, 2048),
}


class ShuffleNetV2(nn.Module):
    def __init__(self, classes, mode='x1_0'):
        super(ShuffleNetV2, self).__init__()
        self.name = f'ShuffleNetV2_{mode}'

        input_channels = 3
        self.repeats = (4, 8, 4)
        self.channels = Config[self.name]

        output_channels = self.channels[0]
        self.conv1_pool = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        input_channels = output_channels

        for i, (repeat, output_channels) in enumerate(zip(self.repeats, self.channels[1:-1])):
            layers = [Unit(input_channels, output_channels, stride=2)]
            for _ in range(repeat - 1):
                layers.append(Unit(output_channels, output_channels, 1))
            setattr(self, f'stage{i + 2}', nn.Sequential(*layers))
            input_channels = output_channels

        output_channels = self.channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(output_channels, classes)
        )

    def forward(self, x):
        x = self.conv1_pool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        return self.fc(x)
