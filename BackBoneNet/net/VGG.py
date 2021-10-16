from torch import nn
from Tools.utils import get_out_layer

# https://arxiv.org/abs/1409.1556
vgg_16_layer = [(2, 64), (2, 128), (3, 256), (3, 512), (3, 512)]
vgg_19_layer = [(2, 64), (2, 128), (4, 256), (4, 512), (4, 512)]


def get_vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers += [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


class VGG(nn.Module):
    def __init__(self, classes, mode=16, shape=(224, 224)):
        super(VGG, self).__init__()
        self.name = f'vgg_{mode}'
        self.layer_list = globals()[f'vgg_{mode}_layer']

        in_channels = 3
        for i, (num_convs, out_channels) in enumerate(self.layer_list):
            setattr(self, f'conv{i}', get_vgg_block(num_convs, in_channels, out_channels))
            in_channels = out_channels

        out = get_out_layer(nn.Sequential(*[getattr(self, f'conv{i}') for i in range(len(self.layer_list))]), 3, shape)

        self.fc = nn.Sequential(nn.Flatten(),
                                nn.Linear(self.layer_list[-1][-1] * out, 4096), nn.ReLU(inplace=True),
                                nn.Dropout(p=0.5),
                                nn.Linear(4096, 4096), nn.ReLU(inplace=True), nn.Dropout(p=0.5),
                                nn.Linear(4096, classes))

    def forward(self, x):
        for i in range(len(self.layer_list)):
            x = getattr(self, f'conv{i}')(x)
        return self.fc(x)
