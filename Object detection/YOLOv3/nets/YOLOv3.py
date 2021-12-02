import torch
from torch import nn
from nets.BackBone import DarkNet53
from torch.nn import functional as F
from nets.block import YOLOv3Loss, YOLOPredict
from utils.config import Config


class YOLOv3(nn.Module):

    def __init__(self):
        super(YOLOv3, self).__init__()
        # out
        self.num_anchors = len(Config['Anchors'])
        self.num_classes = len(Config['Classes'])
        # 主干网络
        self.backbone = DarkNet53()

        self.loss = YOLOv3Loss()
        self.predict = YOLOPredict()

    def forward(self, x, targets=None, device='cpu'):
        x = x.to(device)
        if targets is not None:
            targets = targets.to(device)

        pred = self.backbone(x)
        if self.training:
            loss, loss_items = self.loss(pred, targets, device)
            return loss
        else:
            return self.predict(pred, device)
