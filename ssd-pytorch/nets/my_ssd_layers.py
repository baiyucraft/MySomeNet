from itertools import product as product
from math import sqrt

import torch
from torch import nn
from torch.autograd import Function
from utils.box_utils import decode, nms
from utils.config import Config


class Detect(nn.Module):
    """
    Args:
        num_classes: 种类
        bkg_label: 背景的标签号
        top_k:
        conf_thresh:
        nms_thresh:
    """

    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        """21, 0, 200. 0.5, 0.45"""
        super().__init__()
        self.num_classes = num_classes
        self.background_label = bkg_label

        self.top_k = top_k
        self.conf_thresh = conf_thresh
        self.conf_thresh = 0.1
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        # 0.1，0.2， 将损失放大，trick
        self.variance = Config['variance']

    def forward(self, loc_data, conf_data, prior_data):
        """
        loc_data: 位置置信
        conf_data: 类别置信
        prior_data: 锚框数据
        """
        loc_data = loc_data.cpu().detach()
        conf_data = conf_data.cpu().detach()
        prior_data = prior_data.cpu().detach()

        # batch_size
        batch_size = loc_data.shape[0]
        # 锚框数量
        num_priors = prior_data.shape[0]
        # batch_size * num_classes *  * 5
        output = torch.zeros(batch_size, self.num_classes, self.top_k, 5)

        # reshape：batch_size  * num_classes * num_anchors
        conf_preds = conf_data.permute(0, 2, 1)

        # 对每一张图片进行处理正常预测的时候只有一张图片，所以只会循环一次
        for i in range(batch_size):
            # 获得基于默认框的预测框，b_box形式
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            # 类别置信 num_classes * num_anchors
            conf_scores = conf_preds[i].clone()

            # 查看针对每个类 在 所有框中的 置信分数
            for cl in range(1, self.num_classes):
                # 与 conf_thresh 比较，大于则为True，小于等于为False
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                # 如果没有满足要求的，则进入下一个类
                if not scores.shape[0]:
                    continue

                # 取出满足 conf_thresh 的预测框
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].reshape(-1, 4)

                # 利用这些预测框进行非极大抑制
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1), boxes[ids[:count]]), 1)
                break

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
    in_height, in_width = data[0], data[1]
    device, num_sizes, num_ratios = 'cpu', len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = torch.tensor(sizes, device=device)
    ratio_tensor = torch.tensor(ratios, device=device)

    # 将锚点移动到像素的中心，设置偏移量。因为一个像素的的高为1且宽为1，我们选择偏移我们的中心0.5
    offset_h, offset_w = 0.5, 0.5

    # 生成锚框的所有网格
    center_h = (torch.arange(in_height, device=device) + offset_h) / in_height
    center_w = (torch.arange(in_width, device=device) + offset_w) / in_width
    shift_y, shift_x = torch.meshgrid(center_h, center_w)
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    # 生成锚框
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

    """
    # 生成“boxes_per_pixel”个高和宽，
    # 之后用于创建锚框的四角坐标 (xmin, xmax, ymin, ymax)
    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                   sizes[0] * torch.sqrt(ratio_tensor[1:]))) * in_height / in_width
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
                   sizes[0] / torch.sqrt(ratio_tensor[1:])))
    # 除以2来获得半高和半宽
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(in_height * in_width, 1) / 2

    # 每个中心点都将有“boxes_per_pixel”个锚框，
    # 所以生成含所有锚框中心的网格，重复了“boxes_per_pixel”次
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y],
                           dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    output = out_grid + anchor_manipulations
    return output.unsqueeze(0)
    """
    return output


class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()

        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(n_channels))
        # [scale] * n_channels
        nn.init.constant_(self.weight, scale)

    def reset_parameters(self):
        nn.init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.norm(p=2, dim=1, keepdim=True) + self.eps
        # x /= norm
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out
