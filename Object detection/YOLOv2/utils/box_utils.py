import torch


def b_box_to_c_box(boxes):
    """(x1, y1, x2, y2) to (cx, cy, w, h)"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack((cx, cy, w, h), dim=-1)


def c_box_to_b_box(boxes, d=1):
    """(cx, cy, w, h) to (x1, y1, x2, y2)"""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx / d - 0.5 * w
    y1 = cy / d - 0.5 * h
    x2 = cx / d + 0.5 * w
    y2 = cy / d + 0.5 * h
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


def get_anchors_iou(anchors, boxes):
    boxes_w = boxes[:, 2] - boxes[:, 0] + 1
    boxes_h = boxes[:, 3] - boxes[:, 1] + 1
    iw = torch.min(anchors[:, None, 0], boxes_w)
    ih = torch.min(anchors[:, None, 1], boxes_h)
    inter = iw * ih

    anchors_area = anchors[:, 0] * anchors[:, 1]
    boxes_area = boxes_w * boxes_h
    return inter / (anchors_area[:, None] + boxes_area[None, :] - inter)


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
