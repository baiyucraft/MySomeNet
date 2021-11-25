import torch
from torch import nn
from torch.nn import functional as F
from utils.box_utils import b_box_to_c_box, c_box_to_b_box, box_iou, clip_boxes_to_image, get_anchors_iou
from utils.config import Config
from torchvision.ops import nms


class Reorg(nn.Module):
    def __init__(self, stride=2):
        super(Reorg, self).__init__()
        self.stride = stride

    def forward(self, x):
        batch_size, c, h, w = x.shape
        stride = self.stride

        x = x.reshape(batch_size, c, h // stride, stride, w // stride, stride).transpose(3, 4)
        x = x.reshape(batch_size, c, h // stride * w // stride, stride * stride).transpose(2, 3)
        x = x.reshape(batch_size, c, stride * stride, h // stride, w // stride).transpose(1, 2)
        x = x.reshape(batch_size, stride * stride * c, h // stride, w // stride)
        return x


class YOLOv2Loss(nn.Module):
    def __init__(self):
        super(YOLOv2Loss, self).__init__()
        self.input_size = Config['Size']
        self.w = int(self.input_size[0] / 32)
        self.h = int(self.input_size[1] / 32)
        # 类别
        self.num_anchors = len(Config['Anchors'])
        self.num_classes = len(Config['Classes'])
        self.anchors = Config['Anchors']
        # thresh
        self.iou_thresh = 0.6
        # scale
        self.object_scale = 5.
        self.noobject_scale = 1.
        self.coord_scale = 1.
        self.class_scale = 1.

    def forward(self, bbox_pred, iou_pred, score_pred, targets, device):
        S = self.w
        hw = S * S
        batch_size = bbox_pred.shape[0]
        anchors = torch.Tensor(self.anchors) / 17
        anchors = anchors.to(device)

        # 坐标框
        c_match = torch.meshgrid(torch.arange(S, device=device), torch.arange(S, device=device))
        c_map = torch.stack(c_match, dim=2).unsqueeze(0).unsqueeze(3)

        # 将pred_box转换为真实的bbox
        bbox_pred = bbox_pred.reshape(batch_size, S, S, self.num_anchors, 4)
        c_box_xy = (bbox_pred[:, :, :, :, :2] + c_map).reshape(-1, 2) / S
        c_box_wh = bbox_pred[:, :, :, :, 2:].reshape(-1, self.num_anchors, 2)
        c_box_wh = (c_box_wh * (anchors.unsqueeze(0)) / S).reshape(-1, 2)
        c_box = torch.cat((c_box_xy, c_box_wh), dim=1)
        # 转换后的预测框
        c_box_pred = c_box.reshape(batch_size, -1, self.num_anchors, 4)
        # 输出的预测框 offset
        bbox_pred = bbox_pred.reshape(batch_size, -1, self.num_anchors, 4)

        box_loss, iou_loss, cls_loss = [], [], []
        for i in range(batch_size):
            # 预存容器
            _class = torch.zeros([hw, self.num_anchors, self.num_classes], device=device)
            _class_mask = torch.zeros([hw, self.num_anchors, 1], device=device)

            _iou = torch.zeros([hw, self.num_anchors, 1], device=device)
            _iou_mask = torch.zeros([hw, self.num_anchors, 1], device=device)

            _boxes = torch.zeros([hw, self.num_anchors, 4], device=device)
            _boxes[:, :, 0:2] = 0.5
            _boxes[:, :, 2:4] = 1.0
            _box_mask = torch.zeros([hw, self.num_anchors, 1], device=device) + 0.01

            # 预测值
            c_box_p = c_box_pred[i].reshape(-1, 4)
            iou_p = iou_pred[i]
            # 处理 target
            idx = targets[:, 0] == i
            target_i = targets[idx][:, 1:]
            gt_box = target_i[:, 1:]
            gt_class = target_i[:, 0].long()

            # 处理box
            cell = torch.zeros(size=(S, S), device=device)
            # 转换后的真实框 offset
            tar_gt_box = b_box_to_c_box(gt_box)
            c_xy = tar_gt_box[:, :2] * S
            c_ij = c_xy.floor().long()
            offset_xy = c_xy - c_ij
            tar_gt_box[:, :2] = offset_xy
            # 保留的box
            gt_idx = torch.zeros(gt_box.shape[0], device=device)
            for n, (c_i, c_j) in enumerate(c_ij):
                if cell[c_i, c_j] != 1:
                    cell[c_i, c_j] = 1
                    gt_idx[n] = 1
            gt_box = gt_box[gt_idx == 1]
            tar_gt_box = tar_gt_box[gt_idx == 1]
            gt_class = gt_class[gt_idx == 1]
            len_gt_box = gt_box.shape[0]

            # iou < 0.6
            iou = box_iou(c_box_to_b_box(c_box_p), gt_box)
            best_iou, _ = iou.max(axis=1)
            best_iou = best_iou.reshape(iou_p.shape)
            iou_penalty = 0 - iou_p[best_iou < self.iou_thresh]
            _iou_mask[best_iou <= self.iou_thresh] = self.noobject_scale * iou_penalty

            anchor_iou = get_anchors_iou(anchors, gt_box)
            # 每个真实框对应哪个尺度的anchor
            anchor_index = anchor_iou.argmax(axis=0)

            coo_idx = (cell > 0).flatten()
            anchor_index = anchor_index

            # 对应格子的，对应预测框的，对应真实框的
            # 。。这种方法每格只有一个目标
            # iou
            iou = iou.reshape(hw, self.num_anchors, -1)
            # print(coo_idx.shape, anchor_index.shape)
            # print(_iou_mask.shape, _iou_mask[coo_idx].shape, iou_p.shape, iou_p[coo_idx].shape, '\n')
            _iou_mask[coo_idx, anchor_index] = self.object_scale * (1 - iou_p[coo_idx, anchor_index])
            _iou[coo_idx, anchor_index] = iou[coo_idx, anchor_index, torch.arange(len_gt_box)].unsqueeze(1)

            _box_mask[coo_idx, anchor_index, :] = self.coord_scale
            _boxes[coo_idx, anchor_index, :] = tar_gt_box

            _class_mask[coo_idx, anchor_index, :] = self.class_scale
            _class[coo_idx, anchor_index, gt_class] = 1.

            box_loss.append(F.mse_loss(bbox_pred[i] * _box_mask, _boxes * _box_mask, reduction='sum') / len_gt_box)
            iou_loss.append(F.mse_loss(iou_pred[i] * _iou_mask, _iou * _iou_mask, reduction='sum') / len_gt_box)
            cls_loss.append(F.mse_loss(score_pred[i] * _class_mask, _class * _class_mask, reduction='sum') / len_gt_box)

        return sum(box_loss) + sum(iou_loss) + sum(cls_loss)


class YOLOPredict(nn.Module):
    def __init__(self, confidence=0.1, iou_threshold=0.5):
        super(YOLOPredict, self).__init__()
        self.input_size = Config['Size']
        self.w = int(self.input_size[0] / 32)
        self.h = int(self.input_size[1] / 32)

        self.num_anchors = len(Config['Anchors'])
        self.num_classes = len(Config['Classes'])
        self.anchors = Config['Anchors']
        self.confidence = confidence
        self.iou_threshold = iou_threshold

    def forward(self, bbox_pred, iou_pred, score_pred, device):
        S = self.h
        anchors = torch.Tensor(self.anchors) / 17
        anchors = anchors.to(device)

        box_pred = bbox_pred.reshape(S, S, -1, 4)
        iou_pred = iou_pred.reshape(-1, 1)
        score_pred = score_pred.reshape(-1, self.num_classes)

        # 坐标框 offset_xy 为 c_xy
        c_match = torch.meshgrid(torch.arange(S, device=device), torch.arange(S, device=device))
        c_map = torch.stack(c_match, dim=2).unsqueeze(2)
        c_box_xy = (box_pred[:, :, :, :2] + c_map).reshape(-1, 2) / S
        c_box_wh = box_pred[:, :, :, 2:].reshape(-1, self.num_anchors, 2)
        c_box_wh = (c_box_wh * (anchors.unsqueeze(0)) / S).reshape(-1, 2)
        box_pred = torch.cat((c_box_xy, c_box_wh), dim=1)

        output = []

        # 针对每个图片的每个框 计算 nms
        p_boxes = []
        p_labels = []
        p_scores = []
        boxes = box_pred
        ious = iou_pred
        scores = score_pred
        c_boxes = c_box_to_b_box(boxes)

        # 将框限制在 0-1
        c_boxes = clip_boxes_to_image(c_boxes, (1, 1))
        for cl in range(self.num_classes):
            # 满足最低阈值
            cl_idx = scores[:, cl] > self.confidence
            if cl_idx.sum() == 0:
                continue
            # nms
            cl_box = c_boxes[cl_idx]
            _ = ious[cl_idx,0]
            __=scores[cl_idx, cl]
            cl_scores = ious[cl_idx,0] * scores[cl_idx, cl]
            keep = nms(cl_box, cl_scores, self.iou_threshold)

            p_boxes.append(cl_box[keep])
            p_labels.append(torch.full(keep.shape, cl, device=device))
            p_scores.append(cl_scores[keep])

        if not p_boxes:
            return []
        p_boxes = torch.cat(p_boxes)
        p_labels = torch.cat(p_labels)
        p_scores = torch.cat(p_scores)
        output.append({'boxes': p_boxes,
                       'labels': p_labels,
                       'scores': p_scores})

        return output
