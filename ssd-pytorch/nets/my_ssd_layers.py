from itertools import product as product
from math import sqrt as sqrt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Function
from utils.box_utils import decode, nms
from utils.config import Config


class Detect(Function):
    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        self.nms_thresh = nms_thresh9
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.variance = Config['variance']

    def forward(self, loc_data, conf_data, prior_data):
        # --------------------------------#
        #   先转换成cpu下运行
        # --------------------------------#
        loc_data = loc_data.cpu()
        conf_data = conf_data.cpu()

        # --------------------------------#
        #   num的值为batch_size
        #   num_priors为先验框的数量
        # --------------------------------#
        num = loc_data.size(0)
        num_priors = prior_data.size(0)

        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        # --------------------------------------#
        #   对分类预测结果进行reshape
        #   num, num_classes, num_priors
        # --------------------------------------#
        conf_preds = conf_data.view(num, num_priors, self.num_classes).transpose(2, 1)

        # 对每一张图片进行处理正常预测的时候只有一张图片，所以只会循环一次
        for i in range(num):
            # --------------------------------------#
            #   对先验框解码获得预测框
            #   解码后，获得的结果的shape为
            #   num_priors, 4
            # --------------------------------------#
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            conf_scores = conf_preds[i].clone()

            # --------------------------------------#
            #   获得每一个类对应的分类结果
            #   num_priors,
            # --------------------------------------#
            for cl in range(1, self.num_classes):
                # --------------------------------------#
                #   首先利用门限进行判断
                #   然后取出满足门限的得分
                # --------------------------------------#
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                # --------------------------------------#
                #   将满足门限的预测框取出来
                # --------------------------------------#
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # --------------------------------------#
                #   利用这些预测框进行非极大抑制
                # --------------------------------------#
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1), boxes[ids[:count]]), 1)

        return output


class PriorBox(object):
    def __init__(self, feature_maps, cfg):
        super(PriorBox, self).__init__()
        self.feature_maps = feature_maps
        # 大小
        self.sizes = cfg['sizes']
        # 宽高比
        self.ratios = cfg['ratios']
        
        self.variance = cfg['variance']
        self.clip = cfg['clip']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        output = torch.cat([multibox_prior(f, self.sizes[i], self.ratios[i]) for i, f in enumerate(self.feature_maps)])

        if self.clip:
            output.clamp_(max=1, min=0)
        return output


def multibox_prior(data, sizes, ratios):
    """生成以每个像素为中心具有不同形状的锚框。"""
    in_height, in_width = data[0],data[1]
    device, num_sizes, num_ratios = 'cpu', len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = torch.tensor(sizes, device=device)
    ratio_tensor = torch.tensor(ratios, device=device)

    # 为了将锚点移动到像素的中心，需要设置偏移量。
    # 因为一个像素的的高为1且宽为1，我们选择偏移我们的中心0.5
    offset_h, offset_w = 0.5, 0.5
 
    # 生成锚框的所有网格
    center_h = (torch.arange(in_height, device=device) + offset_h) / in_height
    center_w = (torch.arange(in_width, device=device) + offset_w) / in_width
    shift_y, shift_x = torch.meshgrid(center_h, center_w)
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    # 生成“boxes_per_pixel”个高和宽，
    # 之后用于创建锚框的四角坐标 (xmin, xmax, ymin, ymax)
    # w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
    #                sizes[0] * torch.sqrt(ratio_tensor[1:]))) * in_height / in_width
    # h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
    #                sizes[0] / torch.sqrt(ratio_tensor[1:])))
    # 除以2来获得半高和半宽
    # anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(in_height * in_width, 1) / 2

    # 每个中心点都将有“boxes_per_pixel”个锚框，
    # 所以生成含所有锚框中心的网格，重复了“boxes_per_pixel”次
    # out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y],
    #                        dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    # output = out_grid + anchor_manipulations
    # return output.unsqueeze(0)
    mean = []
    for cx, cy in zip(shift_x, shift_y):
        # 小正方形
        mean += [cx, cy, sizes[0], sizes[0]]
        # 大正方形
        mean += [cx, cy, sizes[1], sizes[1]]
        # 宽高比的正方形
        for r in ratios:
            mean += [cx, cy, sizes[0] * sqrt(r), sizes[0] / sqrt(r)]
            mean += [cx, cy, sizes[0] / sqrt(r), sizes[0] * sqrt(r)]
    output = torch.Tensor(mean).reshape(-1, 4)
    return output


class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        # x /= norm
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out
