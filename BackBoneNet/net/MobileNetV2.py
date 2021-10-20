from torch import nn


def make_divisible(channels, round_value=8, min_value=8):
    new_channels = max(min_value, int(channels + round_value / 2) // round_value * round_value)
    # Make sure that round down does not go down by more than 10%.
    if new_channels < 0.9 * channels:
        new_channels += round_value
    return new_channels


class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1):
        super(BaseConv, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True))

    def forward(self, x):
        return self.conv(x)


class InvertedResidual(nn.Module):
    """Figure 3"""

    def __init__(self, in_channels, out_channels, stride, expand_t):
        super(InvertedResidual, self).__init__()

        hidden_dim = in_channels * expand_t
        self.use_res = stride == 1 and in_channels == out_channels

        self.conv = nn.Sequential(BaseConv(in_channels, hidden_dim, kernel_size=1),
                                  BaseConv(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
                                  nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
                                  nn.BatchNorm2d(out_channels))

    def forward(self, x):
        if self.use_res:
            return x + self.conv(x)
        else:
            return self.conv(x)


config = ((1, 16, 1, 1),
          (6, 24, 2, 2),
          (6, 32, 3, 2),
          (6, 64, 4, 2),
          (6, 96, 3, 1),
          (6, 160, 3, 2),
          (6, 320, 1, 1))


class MobileNetV2(nn.Module):
    def __init__(self, classes, width_mult=1.0):
        super(MobileNetV2, self).__init__()
        self.name = 'MobileNetV2' if width_mult == 1.0 else f'MobileNetV2{width_mult}'
        input_channel = 32
        last_channel = 1280

        input_channel = make_divisible(input_channel * width_mult)
        self.last_channel = make_divisible(last_channel * max(1.0, width_mult))

        self.conv_first = BaseConv(3, input_channel, stride=2)

        self.conv_main = nn.Sequential()
        for i, (t, c, n, s) in enumerate(config):
            output_channel = make_divisible(c * width_mult)
            for j in range(n):
                stride = s if j == 0 else 1
                self.conv_main.add_module(f'bottleneck{i + 1}_{j + 1}',
                                          InvertedResidual(input_channel, output_channel, stride, expand_t=t))
                input_channel = output_channel

        self.conv_last = BaseConv(input_channel, self.last_channel, kernel_size=1)

        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                nn.Flatten(), nn.Dropout(0.2),
                                nn.Linear(self.last_channel, classes))

    def forward(self, x):
        x = self.conv_first(x)
        x = self.conv_main(x)
        x = self.conv_last(x)
        return self.fc(x)
