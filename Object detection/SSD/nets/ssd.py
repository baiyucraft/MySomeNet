import torch
import torch.nn as nn
from utils.config import Config
from nets.all_blocks import Detect, PriorBox, L2Norm

conv_arch = [(2, 64), (2, 128), (3, 256), (3, 512), (3, 512)]


def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
    return nn.Sequential(*layers)


def get_vgg_layer():
    layers = nn.Sequential()
    in_channels = 3
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
    in_channels = layers[-1][-2].out_channels

    # conv8_2   1024,19,19 -> 512,10,10
    conv8_2 = nn.Sequential(nn.Conv2d(in_channels, 256, kernel_size=1, stride=1),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                            nn.ReLU(inplace=True))
    # conv9_2   512,10,10 -> 256,5,5
    conv9_2 = nn.Sequential(nn.Conv2d(512, 128, kernel_size=1, stride=1),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                            nn.ReLU(inplace=True))
    # conv10_2  256,5,5 -> 256,3,3
    conv10_2 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1, stride=1),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(128, 256, kernel_size=3, stride=1),
                             nn.ReLU(inplace=True))
    # conv11_2  256,3,3 -> 256,1,1
    conv11_2 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1, stride=1),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(128, 256, kernel_size=3, stride=1),
                             nn.ReLU(inplace=True))
    layers.add_module('conv8_2', conv8_2)
    layers.add_module('conv9_2', conv9_2)
    layers.add_module('conv10_2', conv10_2)
    layers.add_module('conv11_2', conv11_2)
    return layers


class SSD(nn.Module):
    """
    Args:
        mode: train or test
        net: all net
        loc_layers: 框的位置偏置预测
        conf_layers: 框的类别预测
        feature_maps: 输出的特征图
        num_classes: 预测的类别 + 1
        confidence: 类别的置信阈值
        nms_iou: 框交并比的阈值
    """

    def __init__(self, mode, net, loc_layers, conf_layers, feature_maps, num_classes, confidence, nms_iou):
        super(SSD, self).__init__()
        self.num_classes = num_classes
        self.net = net
        self.loc = loc_layers
        self.conf = conf_layers

        # 类似L2正则化，trick
        self.L2Norm = L2Norm(512, 20)

        # 模式
        self.mode = mode
        if mode == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, confidence, nms_iou)

        self.cfg = Config
        # 生成锚框
        self.priorbox = PriorBox(feature_maps, self.cfg)
        with torch.no_grad():
            self.priors = self.priorbox.forward()

    def forward(self, x):
        batch_size = x.shape[0]
        sources, loc, conf = [], [], []

        # 运行模型，并加入计算值
        for i, layer in enumerate(self.net):
            # conv4_3 加入
            if i == 3:
                for j, lay in enumerate(layer):
                    x = lay(x)
                    if j == 5:
                        sources.append(self.L2Norm(x))
            # conv7 conv8_2 conv9_2 conv_10_2 conv11_2
            elif i >= 6:
                x = layer(x)
                sources.append(x)
            else:
                x = layer(x)

        # 为获得的6个有效特征层添加回归预测和分类预测
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).flatten(start_dim=1))
            conf.append(c(x).permute(0, 2, 3, 1).flatten(start_dim=1))

        # loc  reshape到 batch_size * num_anchors * 4
        # conf  reshape到 batch_size * num_anchors * num_classes
        loc = torch.cat(loc, 1).reshape(batch_size, -1, 4)
        conf = torch.cat(conf, 1).reshape(batch_size, -1, self.num_classes)

        # 如果用于预测的话，会添加上detect用于对先验框解码，获得预测结果
        if self.mode == 'test':
            output = self.detect(loc, self.softmax(conf), self.priors)
        # 不用于预测的话，直接返回网络的回归预测结果和分类预测结果用于训练
        else:
            output = (loc, conf, self.priors)
        return output


def get_multibox(layers, num_classes):
    """Multibox"""
    loc_layers = nn.ModuleList()
    conf_layers = nn.ModuleList()

    feature_maps = [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]

    # conv4_3 to (512,38,38)
    in_channels = layers[3][4].out_channels
    loc_layers.add_module('conv4_3_loc', nn.Conv2d(in_channels, 4 * 4, kernel_size=3, padding=1))
    conf_layers.add_module('conv4_3_conf', nn.Conv2d(in_channels, 4 * num_classes, kernel_size=3, padding=1))

    # conv7 to conv7(1024,19,19)
    in_channels = layers[6][-2].out_channels
    loc_layers.add_module('conv7_loc', nn.Conv2d(in_channels, 6 * 4, kernel_size=3, padding=1))
    conf_layers.add_module('conv7_conf', nn.Conv2d(in_channels, 6 * num_classes, kernel_size=3, padding=1))

    # conv8 to (512,10,10)
    in_channels = layers[7][-2].out_channels
    loc_layers.add_module('conv8_loc', nn.Conv2d(in_channels, 6 * 4, kernel_size=3, padding=1))
    conf_layers.add_module('conv8_conf', nn.Conv2d(in_channels, 6 * num_classes, kernel_size=3, padding=1))

    # conv9 to (256,5,5)
    in_channels = layers[8][-2].out_channels
    loc_layers.add_module('conv9_loc', nn.Conv2d(in_channels, 6 * 4, kernel_size=3, padding=1))
    conf_layers.add_module('conv9_conf', nn.Conv2d(in_channels, 6 * num_classes, kernel_size=3, padding=1))

    # conv10 to (256,3,3)
    in_channels = layers[9][-2].out_channels
    loc_layers.add_module('conv10_loc', nn.Conv2d(in_channels, 4 * 4, kernel_size=3, padding=1))
    conf_layers.add_module('conv10_conf', nn.Conv2d(in_channels, 4 * num_classes, kernel_size=3, padding=1))

    # conv11 to (256,1,1)
    in_channels = layers[10][-2].out_channels
    loc_layers.add_module('conv11_loc', nn.Conv2d(in_channels, 4 * 4, kernel_size=3, padding=1))
    conf_layers.add_module('conv11_conf', nn.Conv2d(in_channels, 4 * num_classes, kernel_size=3, padding=1))
    return loc_layers, conf_layers, feature_maps


def get_ssd(mode, num_classes, confidence=0.5, nms_iou=0.45):
    """
    Args:
        mode: train or test
        num_classes: 识别种类
        confidence: 类别的置信阈值
        nms_iou: 框交并比的阈值
    """
    # 主干网络和额外网络
    layers = get_vgg_layer()
    # Multibox, loc_layers: bbox_predict, conf_layers: class_predict,
    loc_layers, conf_layers, feature_maps = get_multibox(layers, num_classes)
    ssd = SSD(mode, layers, loc_layers, conf_layers, feature_maps, num_classes, confidence, nms_iou)
    return ssd


def get_layer(ssd, x):
    for i in ssd.net:
        x = i(x)
        print(x.shape)


if __name__ == '__main__':
    ssd = get_ssd('test', 3)
    x = torch.randn(size=(1, 3, 300, 300))
    # print(ssd)
    get_layer(ssd, x)
    # print(ssd(x).shape)
