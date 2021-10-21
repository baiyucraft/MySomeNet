from torch import nn


# https://arxiv.org/abs/1905.02244
def make_divisible(channels, round_value=8, min_value=8):
    new_channels = max(min_value, int(channels + round_value / 2) // round_value * round_value)
    # Make sure that round down does not go down by more than 10%.
    if new_channels < 0.9 * channels:
        new_channels += round_value
    return new_channels


class SqueezeExcitation(nn.Module):
    """SE block"""

    def __init__(self, in_channels, squeeze_=4):
        super(SqueezeExcitation, self).__init__()
        squeeze_channels = make_divisible(in_channels // squeeze_)
        self.se_block = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                      nn.Conv2d(in_channels, squeeze_channels, kernel_size=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(squeeze_channels, in_channels, kernel_size=1),
                                      nn.Hardsigmoid(inplace=True))

    def forward(self, x):
        return self.se_block(x) * x


class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, ac_layer=None):
        super(BaseConv, self).__init__()
        padding = (kernel_size - 1) // 2
        if ac_layer is None:
            ac_layer = nn.ReLU6

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            ac_layer(inplace=True))

    def forward(self, x):
        return self.conv(x)


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, expanded_channels, kernel, stride, use_se, use_hs):
        super(InvertedResidual, self).__init__()
        self.use_res = stride == 1 and in_channels == out_channels
        ac_layer = nn.Hardswish if use_hs == 'HS' else nn.ReLU

        layers = []
        # expand
        if expanded_channels != in_channels:
            layers.append(BaseConv(in_channels, expanded_channels, kernel_size=1, ac_layer=ac_layer))

        # dw
        layers.append(
            BaseConv(expanded_channels, expanded_channels, kernel_size=kernel, stride=stride, groups=expanded_channels,
                     ac_layer=ac_layer))
        if use_se:
            layers.append(SqueezeExcitation(expanded_channels))

        # pro
        layers.append(BaseConv(expanded_channels, out_channels, kernel_size=1, ac_layer=ac_layer))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res:
            return x + self.block(x)
        else:
            return self.block(x)


config = {
    'MobileNetV3_large': (
        (3, 16, 16, False, 'RE', 1),
        (3, 64, 24, False, 'RE', 2),
        (3, 72, 24, False, 'RE', 1),
        (5, 72, 40, True, 'RE', 2),
        (5, 120, 40, True, 'RE', 1),
        (5, 120, 40, True, 'RE', 1),
        (3, 240, 80, False, 'HS', 2),
        (3, 200, 80, False, 'HS', 1),
        (3, 184, 80, False, 'HS', 1),
        (3, 184, 80, False, 'HS', 1),
        (3, 480, 112, True, 'HS', 1),
        (3, 672, 112, True, 'HS', 1),
        (5, 672, 160, True, 'HS', 2),
        (5, 960, 160, True, 'HS', 1),
        (5, 960, 160, True, 'HS', 1),
        1280
    ),
    'MobileNetV3_small': (
        (3, 16, 16, True, 'RE', 2),
        (3, 72, 24, False, 'RE', 2),
        (3, 88, 24, False, 'RE', 1),
        (5, 96, 40, True, 'HS', 2),
        (5, 240, 40, True, 'HS', 1),
        (5, 240, 40, True, 'HS', 1),
        (5, 120, 48, True, 'HS', 1),
        (5, 144, 48, True, 'HS', 1),
        (5, 288, 96, True, 'HS', 2),
        (5, 576, 96, True, 'HS', 1),
        (5, 576, 96, True, 'HS', 1),
        1024
    )
}


class MobileNetV3(nn.Module):
    def __init__(self, classes, width_mult=1.0, mode='large'):
        super(MobileNetV3, self).__init__()
        self.name = f'MobileNetV3_{mode}'
        self.model_setting = config[self.name][:-1]
        input_channel = 16
        last_channel = config[self.name][-1]

        input_channel = make_divisible(input_channel * width_mult)
        self.last_channel = make_divisible(last_channel * max(1.0, width_mult))

        self.conv_first = BaseConv(3, input_channel, kernel_size=3, stride=2, ac_layer=nn.Hardswish)

        self.conv_main = nn.Sequential()
        for i, (k, exp_c, out_c, se, nl, s) in enumerate(self.model_setting):
            expanded_channels = make_divisible(exp_c * width_mult)
            output_channels = make_divisible(out_c * width_mult)
            self.conv_main.add_module(f'bneck{i + 1}',
                                      InvertedResidual(input_channel, output_channels, expanded_channels, kernel=k,
                                                       stride=s, use_se=se, use_hs=nl))
            input_channel = output_channels

        last_channel = make_divisible(input_channel * 6 * width_mult)
        self.conv_last = BaseConv(input_channel, last_channel, kernel_size=1, ac_layer=nn.Hardswish)

        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                nn.Flatten(),
                                nn.Linear(last_channel, self.last_channel),
                                nn.Hardswish(inplace=True),
                                nn.Dropout(p=0.2, inplace=True),
                                nn.Linear(self.last_channel, classes), )

    def forward(self, x):
        x = self.conv_first(x)
        x = self.conv_main(x)
        x = self.conv_last(x)
        return self.fc(x)
