import torch
from torch import nn
from torch.nn import functional as F
from utils.box_utils import b_box_to_c_box, c_box_to_b_box, box_iou, clip_boxes_to_image
from utils.config import Config
from torchvision.ops import nms


class YOLOv1Loss(nn.Module, ):
    def __init__(self, l_coord=5, l_noobj=0.5):
        super(YOLOv1Loss, self).__init__()
        # 将图像分为 S × S
        self.S = Config['S']
        # S × S × (B ∗ 5 + C)
        self.B = Config['B']
        # 类别
        self.num_classes = len(Config['Classes'])
        self.l_coord = l_coord
        self.l_noobj = l_noobj

    def forward(self, pred, targets, device):
        # 批次大小
        batch_size = pred.shape[0]

        # 变换target
        target_list = []
        cell_size = 1. / self.S
        last_id = self.B * 5
        for i in range(batch_size):
            idx = targets[:, 0] == i
            target_i = targets[idx][:, 1:]
            # b_box to c_box
            target_i[:, 1:] = b_box_to_c_box(target_i[:, 1:])

            target_tmp = torch.zeros(size=(self.S, self.S, 5 * self.B + self.num_classes), device=device)
            for l_c_box in target_i:
                label = l_c_box[0].long()
                c_xy = l_c_box[1:3]
                c_wh = l_c_box[3:]
                # 计算在 S × S 中的哪格
                c_ij = ((c_xy / cell_size).ceil() - 1).long()
                offset_xy = (c_xy - c_ij * cell_size) / cell_size
                # 写入对应 ij
                target_tmp_c = target_tmp[c_ij[0], c_ij[1], :]
                target_tmp_c[0:last_id:5] = offset_xy[0]
                target_tmp_c[1:last_id:5] = offset_xy[1]
                target_tmp_c[2:last_id:5] = c_wh[0]
                target_tmp_c[3:last_id:5] = c_wh[1]
                target_tmp_c[4:last_id:5] = 1
                target_tmp_c[label + last_id] = 1
            target_list.append(target_tmp)
        targets = torch.stack(target_list, dim=0)

        # 具有obj的 格子索引
        coo_idx = targets[:, :, :, 4] > 0
        # 不具有obj的 格子索引
        noo_idx = targets[:, :, :, 4] == 0

        # coo 具有obj的预测框
        coo_pred = pred[coo_idx]
        box_pred = coo_pred[:, :10].reshape(-1, 5)
        class_pred = coo_pred[:, 10:]
        # coo 具有obj的真实框
        coo_target = targets[coo_idx]
        box_target = coo_target[:, :10].reshape(-1, 5)
        class_target = coo_target[:, 10:]

        # 计算具有obj的格子中 选择目标 self.B(2) 中选 1
        box_num = box_pred.shape[0]
        obj_idx = torch.zeros(size=[box_num], device=device).long()
        box_target_iou = torch.zeros(size=[box_num], device=device)
        for i in range(0, box_num, 2):
            box1 = box_pred[i:i + 2]
            box2 = box_target[i, None]
            # 转换 缩小偏置
            box1 = c_box_to_b_box(box1, d=14)
            box2 = c_box_to_b_box(box2, d=14)
            # (2,1)
            iou = box_iou(box1, box2)
            max_iou, max_index = iou.max(0)

            # 选中的 预测框
            obj_idx[i + max_index] = 1

            # 选中的预测框的iou
            box_target_iou[i + max_index] = max_iou

        # 是 obj 的预测框
        box_pred_obj = box_pred[obj_idx == 1]
        # 是 obj 的真实框
        box_target_obj = box_target[obj_idx == 1]
        # 是 obj 的真实框与预测框的 iou, iou为1即为重合
        box_target_obj_iou = box_target_iou[obj_idx == 1]

        # coord: obj损失
        # 1/2: 是 obj 的位置损失
        xy_loss = F.mse_loss(box_pred_obj[:, :2], box_target_obj[:, :2], reduction='sum')
        wh_loss = F.mse_loss(torch.sqrt(box_pred_obj[:, 2:4]), torch.sqrt(box_target_obj[:, 2:4]), reduction='sum')
        loc_loss = xy_loss + wh_loss
        # 3: obj 的box的 confidence 目标损失
        contain_loss = F.mse_loss(box_pred_obj[:, 4], box_target_obj_iou, reduction='sum')
        # 5: 类别损失
        class_loss = F.mse_loss(class_pred, class_target, reduction='sum')

        # noo 中 no_obj 的 confidence 损失
        noo_pred = pred[noo_idx]
        noo_target = targets[noo_idx]
        noo_pred_c = noo_pred[:, 4:last_id:5].flatten()
        noo_target_c = noo_target[:, 4:last_id:5].flatten()
        # 4: 不是obj的box的 confidence 目标损失
        no_obj_loss = F.mse_loss(noo_pred_c, noo_target_c, reduction='sum')
        # print(loc_loss, contain_loss, no_obj_loss, class_loss)
        return (self.l_coord * loc_loss + contain_loss + self.l_noobj * no_obj_loss + class_loss) / batch_size


class YOLOPredict(nn.Module):
    def __init__(self, confidence=0.1, iou_threshold=0.5):
        super(YOLOPredict, self).__init__()
        # 将图像分为 S × S
        self.S = Config['S']
        # S × S × (B ∗ 5 + C)
        self.B = Config['B']
        # 类别
        self.num_classes = len(Config['Classes'])
        self.confidence = confidence
        self.iou_threshold = iou_threshold

    def forward(self, pred, device):
        batch_size = pred.shape[0]
        cell_size = 1. / self.S
        last_id = self.B * 5

        # (batch_size, S, S, 5 * B + num_classes) -> (batch_size, S, S, B, 4 + num_classes)
        # iou_confidence * class_confidence
        pred_box_match = pred[:, :, :, :last_id].reshape(batch_size, self.S, self.S, self.B, 5)
        pred_box = pred_box_match[:, :, :, :, :4]
        pred_box_confidence = pred_box_match[:, :, :, :, 4].unsqueeze(4)
        pred_class_confidence = pred[:, :, :, last_id:].reshape(batch_size, self.S, self.S, 1, self.num_classes)
        pred_conf = pred_class_confidence * pred_box_confidence
        pred_c_box = torch.cat((pred_box, pred_conf), dim=4)

        # 从 offset_xy 转换为 c_xy
        offset_xy = pred_c_box[:, :, :, :, :2]
        c_i, c_j = torch.meshgrid(torch.arange(self.S, device=device), torch.arange(self.S, device=device))
        c_ij = torch.stack((c_i, c_j), dim=2).unsqueeze(0).unsqueeze(3).expand_as(offset_xy)
        pred_c_box[:, :, :, :, :2] = offset_xy * cell_size + c_ij * cell_size

        output = []
        # 针对每个图片的每个框 计算 nms
        pred_change = pred_c_box.reshape(batch_size, self.S * self.S * self.B, -1)
        for pred_img in pred_change:
            p_boxes = []
            p_labels = []
            p_scores = []
            boxes = pred_img[:, :4]
            scores = pred_img[:, 4:]
            c_boxes = c_box_to_b_box(boxes)

            # 将框限制在 0-1
            c_boxes = clip_boxes_to_image(c_boxes, (1, 1))
            # print(c_boxes, scores.shape)
            for cl in range(self.num_classes):
                # 满足最低阈值
                cl_idx = scores[:, cl] > self.confidence
                if cl_idx.sum() == 0:
                    continue
                # nms
                cl_box = c_boxes[cl_idx, :]
                cl_scores = scores[cl_idx, cl]
                keep = nms(cl_box, cl_scores, self.iou_threshold)

                p_boxes.append(cl_box[keep])
                p_labels.append(torch.full(keep.shape, cl, device=device))
                p_scores.append(cl_scores[keep])
            if not p_boxes:
                continue
            p_boxes = torch.cat(p_boxes)
            p_labels = torch.cat(p_labels)
            p_scores = torch.cat(p_scores)
            output.append({'boxes': p_boxes,
                           'labels': p_labels,
                           'scores': p_scores})

        return output
