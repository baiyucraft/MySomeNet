from torch import nn
from nets.BackBone import BackBone
from nets.block import YOLOv1Loss, YOLOPredict
from utils.config import Config


class YOLOv1(nn.Module):
    """
    Args:
        l_coord: 边界框坐标损失 权重
        l_noobj: 没有object的bbox的置信损失 权重
    """

    def __init__(self, l_coord=5, l_noobj=0.5):
        super(YOLOv1, self).__init__()
        # 将图像分为 S × S
        self.S = Config['S']
        # S × S × (B ∗ 5 + C)
        self.B = Config['B']
        # 类别
        self.num_classes = len(Config['Classes'])
        self.l_coord = l_coord
        self.l_noobj = l_noobj
        # 主干网络
        self.backbone = BackBone()
        self.loss = YOLOv1Loss(l_coord, l_noobj)
        self.predict = YOLOPredict()

    def forward(self, x, targets=None, device='cpu'):
        x = x.to(device)
        if targets is not None:
            targets = targets.to(device)
        pred = self.backbone(x)

        if self.training:
            return self.loss(pred, targets, device)
        else:
            return self.predict(pred, device)
