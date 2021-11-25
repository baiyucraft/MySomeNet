import torch
from torch import nn
from nets.BackBone import DarkNet19
from torch.nn import functional as F
from nets.block import YOLOv2Loss, YOLOPredict
from utils.config import Config


class YOLOv2(nn.Module):
    """
    Args:
        l_coord: 边界框坐标损失 权重
        l_noobj: 没有object的bbox的置信损失 权重
    """

    def __init__(self, l_coord=5, l_noobj=0.5):
        super(YOLOv2, self).__init__()
        # out
        self.num_anchors = len(Config['Anchors'])
        self.num_classes = len(Config['Classes'])

        self.l_coord = l_coord
        self.l_noobj = l_noobj
        # 主干网络
        self.backbone = DarkNet19()

        self.loss = YOLOv2Loss()
        self.predict = YOLOPredict()

    def forward(self, x, targets=None, device='cpu'):
        batch_size = x.shape[0]
        x = x.to(device)
        if targets is not None:
            targets = targets.to(device)
        # batch_size * (num_anchors * (num_classes + 5)) * S * S
        pred = self.backbone(x)
        pred = pred.permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_anchors, self.num_classes + 5)

        # tx, ty, tw, th, to -> sig(tx), sig(ty), exp(tw), exp(th), sig(to)
        xy_pred = torch.sigmoid(pred[:, :, :, :2])
        wh_pred = torch.exp(pred[:, :, :, 2:4])
        bbox_pred = torch.cat([xy_pred, wh_pred], 3)
        iou_pred = torch.sigmoid(pred[:, :, :, 4:5])
        score_pred = torch.softmax(pred[:, :, :, 5:], dim=3)

        if self.training:
            return self.loss(bbox_pred, iou_pred, score_pred, targets, device)
        else:
            return self.predict(bbox_pred, iou_pred, score_pred, device)
