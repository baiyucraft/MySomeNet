import math

import torch
from torchvision.ops import nms


def b_box_to_c_box(boxes):
    """(x1, y1, x2, y2) to (cx, cy, w, h)"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack((cx, cy, w, h), dim=-1)


def c_box_to_b_box(boxes):
    """(cx, cy, w, h) to (x1, y1, x2, y2)"""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack((x1, y1, x2, y2), dim=-1)


def box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(box_a, box_b):
    """
    计算交并比
    Args:
        box_a (Tensor[N, 4])
        box_b (Tensor[M, 4])
    """

    areas_a = box_area(box_a)
    areas_b = box_area(box_b)
    # 计算相交的 左上 右下，并计算面积
    inter_left_uppers = torch.max(box_a[:, None, :2], box_b[:, :2])
    inter_right_lowers = torch.min(box_a[:, None, 2:], box_b[:, 2:])
    inters = (inter_right_lowers - inter_left_uppers).clamp(min=0)

    inter = inters[:, :, 0] * inters[:, :, 1]
    union = areas_a[:, None] + areas_b - inter
    return inter / union


def clip_boxes_to_image(boxes, size):
    """
    将框的大小限制在image_size内
    """
    dim = boxes.dim()
    boxes_x = boxes[..., 0::2]
    boxes_y = boxes[..., 1::2]
    height, width = size

    boxes_x = boxes_x.clamp(min=0, max=width)
    boxes_y = boxes_y.clamp(min=0, max=height)

    clipped_boxes = torch.stack((boxes_x, boxes_y), dim=dim)
    return clipped_boxes.reshape(boxes.shape)


def remove_small_boxes(boxes, min_size):
    """
    移除 w、h较小的框
    """
    ws, hs = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    keep = (ws >= min_size) & (hs >= min_size)
    keep = torch.where(keep)[0]
    return keep


def batched_nms(boxes, scores, idxs, iou_threshold):
    """
    Args:
        boxes: 框
        scores: 分数
        idxs: 不同lvl（批次）
        iou_threshold: 阈值
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    else:
        # ！！！根据lvl作偏置, 保证不同批次（5批次）不重叠
        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
        boxes_for_nms = boxes + offsets[:, None]
        # c++实现 非极大抑制
        keep = nms(boxes_for_nms, scores, iou_threshold)
        return keep


class BoxCoder(object):
    def __init__(self, weights, bbox_xform_clip=math.log(1000. / 16)):
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, reference_boxes, proposals):
        """合起来计算然后分开"""
        boxes_per_image = [len(b) for b in reference_boxes]
        reference_boxes = torch.cat(reference_boxes, dim=0)
        proposals = torch.cat(proposals, dim=0)
        targets = self.encode_single(reference_boxes, proposals)
        return targets.split(boxes_per_image, 0)

    def encode_single(self, reference_boxes, proposals):
        """
        Args:
            reference_boxes: 真实框
            proposals: 基于像素的框
        具体计算:
            target_x = w * (b_center_x - p_center_x) / p_width
            target_y = w * (b_center_y - p_center_y) / p_height
            target_w = w * log(b_width / p_width)
            target_h = w * log(b_height / p_height)
        Return:
            回归偏差
        """
        dtype = reference_boxes.dtype
        device = reference_boxes.device
        weights = torch.as_tensor(self.weights, dtype=dtype, device=device)
        wx, wy, ww, wh = weights[0], weights[1], weights[2], weights[3]

        ex = b_box_to_c_box(proposals)
        gt = b_box_to_c_box(reference_boxes)

        targets_xy = wx * (gt[:, :2] - ex[:, :2]) / ex[:, 2:]
        targets_wh = torch.log(gt[:, 2:] / ex[:, 2:])

        targets = torch.cat((targets_xy, targets_wh), dim=1)
        return targets

    def decode(self, rel_codes, boxes):
        # [anchor_num] * batch_size
        boxes_per_image = [b.size(0) for b in boxes]
        concat_boxes = torch.cat(boxes, dim=0)
        box_sum = 0
        for val in boxes_per_image:
            box_sum += val
        if box_sum > 0:
            rel_codes = rel_codes.reshape(box_sum, -1)
        pred_boxes = self.decode_single(rel_codes, concat_boxes)
        if box_sum > 0:
            pred_boxes = pred_boxes.reshape(box_sum, -1, 4)
        return pred_boxes

    def decode_single(self, rel_codes, boxes):
        """
        Args:
            rel_codes: 回归偏差
            boxes: 基于像素的框
        具体计算:
            pred_x = boxes_x + rel_codes_x * boxes_w / w
            pred_y = boxes_y + rel_codes_y * boxes_h / w
            pred_width = boxes_width * exp(rel_codes_w  / w)
            pred_height = boxes_height * exp(rel_codes_h  / w)
        Return:
            真实的预测框
        """
        boxes = boxes.to(rel_codes.dtype)
        boxes = b_box_to_c_box(boxes)

        ctr_x, ctr_y, widths, heights = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

        wx, wy, ww, wh = self.weights

        dx = rel_codes[:, 0::4] / wx
        dy = rel_codes[:, 1::4] / wy
        dw = rel_codes[:, 2::4] / ww
        dh = rel_codes[:, 3::4] / wh
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes1 = pred_ctr_x - torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w
        pred_boxes2 = pred_ctr_y - torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h
        pred_boxes3 = pred_ctr_x + torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w
        pred_boxes4 = pred_ctr_y + torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h

        pred_boxes = torch.stack((pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4), dim=2).flatten(1)

        return pred_boxes


class Matcher(object):
    """
    实现anchor与gt的配对
    """

    def __init__(self, high_threshold=0.7, low_threshold=0.3, allow_low_quality_matches=False):
        """
        每一个anchor都找一个与之iou最大的gt。若 max_iou > high_threshold，则该anchor的label为1，即认定该anchor是目标；
        若max_iou<low_threshold，则该anchor的label为0，即认定该anchor为背景；
        若max_iou介于low_threshold和high_threshold之间，则忽视该anchor，不纳入损失函数。
        gt可对应０个或者多个anchor，anchor可对应0或1个anchor。
        这个匹配操作是基于box_iou返回的iou矩阵进行的。
        Args:
            high_threshold: 大于的为目标
            low_threshold: 小于的为背景，居中不计入
            allow_low_quality_matches: 如果值为真，则允许anchor匹配上小iou的gt，因为可能有一个gt与所有的anchor之间的iou都小于high_threshold
            为了让每个gt都有与之配对的anchor，则配对该 gt 和与之 iou 最大anchor
        Return:
            长度为N的向量，其表示每一个锚点的类型：背景-1,介于背景和目标之间-2以及目标边框（对应最大gt的基准边框的索引）
        """
        self.BELOW_LOW_THRESHOLD = -1
        self.BETWEEN_THRESHOLDS = -2
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix):
        """
        Args:
            match_quality_matrix: M gt x N anchor.

        Returns:
            matches: 每个框对应的值
        """
        # 给定anchor，找与之iou最大的gt，mathed_vals为每列的最大值，mathes为每列最大值的索引
        matched_vals, matches = match_quality_matrix.max(dim=0)
        if self.allow_low_quality_matches:
            all_matches = matches.clone()
        else:
            all_matches = None

        # 选取符合条件的下标
        below_low_threshold = matched_vals < self.low_threshold
        between_thresholds = (matched_vals >= self.low_threshold) & (matched_vals < self.high_threshold)
        matches[below_low_threshold] = self.BELOW_LOW_THRESHOLD
        matches[between_thresholds] = self.BETWEEN_THRESHOLDS

        if self.allow_low_quality_matches:
            assert all_matches is not None
            # 给定gt，与之对应的最大iou的anchor，即便iou小于阈值，也把它作为目标
            self.set_low_quality_matches_(matches, all_matches, match_quality_matrix)

        return matches

    def set_low_quality_matches_(self, matches, all_matches, match_quality_matrix):
        """
        保证每个 gt 有对应 anchor
        """
        # 对每个 gt 交并比最大的anchor
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
        # 索引为 i 行 j 列
        gt_pred_pairs_of_highest_quality = torch.where(match_quality_matrix == highest_quality_foreach_gt[:, None])

        pred_inds_to_update = gt_pred_pairs_of_highest_quality[1]
        matches[pred_inds_to_update] = all_matches[pred_inds_to_update]


class BalancedPositiveNegativeSampler(object):
    """
    通过随机采样平衡正负样本数量，
    正样本采: batch_size_per_image * positive_fraction，
    负样本采: batch_size_per_image  -正样本数
    """

    def __init__(self, batch_size_per_image=256, positive_fraction=0.5):
        # type: (int, float) -> None
        """
        Args:
            batch_size_per_image (int): number of elements to be selected per image
            positive_fraction (float): percentace of positive elements per batch
        """
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction

    def __call__(self, matched_idxs):
        """
        Args:
            matched idxs: 1为正样本 0为负样本
        Returns:
            pos_idx (list[tensor]): 优先选择所有正样本
            neg_idx (list[tensor])
        """
        pos_idx = []
        neg_idx = []
        for matched_idxs_per_image in matched_idxs:
            positive = torch.where(matched_idxs_per_image >= 1)[0]
            negative = torch.where(matched_idxs_per_image == 0)[0]

            num_pos = int(self.batch_size_per_image * self.positive_fraction)
            # 计算正样本数 小于 num_pos 则改为真实值
            num_pos = min(positive.numel(), num_pos)
            num_neg = self.batch_size_per_image - num_pos
            # protect against not enough negative examples
            num_neg = min(negative.numel(), num_neg)

            # 随机选择正负样本的下标
            perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
            perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

            pos_idx_per_image = positive[perm1]
            neg_idx_per_image = negative[perm2]

            # 生成索引mask
            pos_idx_per_image_mask = torch.zeros_like(
                matched_idxs_per_image, dtype=torch.uint8
            )
            neg_idx_per_image_mask = torch.zeros_like(
                matched_idxs_per_image, dtype=torch.uint8
            )

            pos_idx_per_image_mask[pos_idx_per_image] = 1
            neg_idx_per_image_mask[neg_idx_per_image] = 1

            pos_idx.append(pos_idx_per_image_mask)
            neg_idx.append(neg_idx_per_image_mask)

        return pos_idx, neg_idx


def smooth_l1_loss(input, target, beta=1. / 9, size_average=True):
    """
    计算 smooth_l1 损失, 以 1/9 为分界线
    """
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()
