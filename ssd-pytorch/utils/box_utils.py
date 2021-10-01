import numpy as np
import torch
from PIL import Image


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


def get_iou(box_a, box_b):
    """获得两个框的交并比"""

    def get_area(boxes):
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    areas_a = get_area(box_a)
    areas_b = get_area(box_b)
    # 计算相交的 左上 右下，并计算面积
    inter_left_uppers = torch.max(box_a[:, None, :2], box_b[:, :2])
    inter_right_lowers = torch.min(box_a[:, None, 2:], box_b[:, 2:])
    inters = (inter_right_lowers - inter_left_uppers).clamp(min=0)

    inter = inters[:, :, 0] * inters[:, :, 1]
    union = areas_a[:, None] + areas_b - inter
    return inter / union


def match(threshold, truths, priors, variances, labels):
    """
    计算所有 锚框 和 真实框 的重合程度
    Args:
        threshold: 阈值
        truths: 真实框
        priors: 预测锚框
        variances: trick
        labels: 标签
    """
    # 计算交并比
    overlaps = get_iou(truths, c_box_to_b_box(priors))

    # 得到与 每个真实框 重合最好 的锚框，长度为真实框个数
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    best_prior_idx.squeeze_()
    best_prior_overlap.squeeze_()

    # 得到与 每个锚框 重合最好 的真实框，长度为锚框个数
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_()
    best_truth_overlap.squeeze_()

    # 保证每个 真实框 至少对应 一个 锚框
    for j in range(best_prior_idx.shape[0]):
        best_truth_idx[best_prior_idx[j]] = j

    # 给其中填充 2
    best_truth_overlap.index_fill_(dim=0, index=best_prior_idx, value=2)

    # 获取 每一个 锚框 对应的 真实框 的 boxes
    matches = truths[best_truth_idx]
    # 类别
    conf = labels[best_truth_idx] + 1
    # 如果重合度 小于 阈值 则认为是背景
    conf[best_truth_overlap < threshold] = 0
    # 得到偏置
    loc = encode(matches, priors, variances)

    return loc, conf


def encode(matches, priors, variances):
    """
    计算loc(c_box)
    具体计算:
        loc_x = (b_center_x - p_center_x) / (p_width * v)
        loc_y = (b_center_y - p_center_y) / (p_height * v)
        loc_w = log(b_width / p_width) / v
        loc_h = log(b_height / p_height) / v
    """
    matches = b_box_to_c_box(matches)

    g_cxcy = (matches[:, :2] - priors[:, :2]) / (variances[0] * priors[:, 2:])
    g_wh = matches[:, 2:] / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    return torch.cat([g_cxcy, g_wh], 1)


def decode(loc, priors, variances):
    """
    得到预测框(b_box)
    Args:
        loc: 偏移值
        priors: 默认框
        variances: trick
    具体计算:
        b_center_x = p_center_x + loc_x * p_width * v
        b_center_y = p_center_y + loc_y * p_height * v
        b_width = p_width * exp(loc_w  * v)
        b_height = p_height * exp(loc_h  * v)
    """
    boxes = torch.cat((priors[:, :2] + loc[:, :2] * priors[:, 2:] * variances[0],
                       priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    # 将 center_box 转换为 bounding_box(x_min, y_min, x_max, y_max)
    return c_box_to_b_box(boxes)


def nms(boxes, scores, overlap=0.5, top_k=200):
    """
    非极大抑制，筛选出一定区域内得分最大的框
    Return: 保留的 下标 以及 个数
    """
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    # 面积
    area = (x2 - x1) * (y2 - y1)
    # 将分数排序，取大的 top_k 个
    v, idx = scores.sort(0)
    idx = idx[:top_k]

    keep, count = [], 0
    while idx.numel() > 0:
        # 从idx取出score最高的下标，并将此框的面积与剩下的框计算交并比iou
        i = idx[-1]
        keep.append(i)
        count += 1
        if idx.size(0) == 1:
            break

        idx = idx[:-1]
        # 计算相交的面积（交集）
        in_box = boxes[idx]
        xx1, yy1 = in_box[:, 0].clamp(min=x1[i]), in_box[:, 1].clamp(min=y1[i])
        xx2, yy2 = in_box[:, 2].clamp(max=x2[i]), in_box[:, 3].clamp(max=y2[i])
        w = (xx2 - xx1).clamp(0.0)
        h = (yy2 - yy1).clamp(0.0)
        inter = w * h

        # 剩下的框 面积
        rem_areas = area[idx]
        # 整体部分（并集）
        union = (rem_areas - inter) + area[i]
        iou = inter / union
        # 迭代 小于等于 阈值的 idx
        idx = idx[iou <= overlap]
    return torch.Tensor(keep).long(), count


def log_sum_exp(x):
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x - x_max), 1, keepdim=True)) + x_max


def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image


def ssd_correct_boxes(top, left, bottom, right, input_shape, image_shape):
    new_shape = image_shape * np.min(input_shape / image_shape)

    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape

    box_yx = np.concatenate(((top + bottom) / 2, (left + right) / 2), axis=-1)
    box_hw = np.concatenate((bottom - top, right - left), axis=-1)

    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = np.concatenate([
        box_mins[:, 0:1],
        box_mins[:, 1:2],
        box_maxes[:, 0:1],
        box_maxes[:, 1:2]
    ], axis=-1)
    boxes *= np.concatenate([image_shape, image_shape], axis=-1)
    return boxes
