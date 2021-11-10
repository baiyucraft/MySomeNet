import torch
import torch.nn as nn
from nets.FasterRCNNTransform import FasterRCNNTransform
from nets.ROI import MultiScaleRoIAlign, TwoMLPHead, FastRCNNPredictor, RoIHeads
from nets.RPN import AnchorGenerator, RPNHead, RegionProposalNetwork
from nets.BackBone import BackBone


class FasterRCNN(nn.Module):
    def __init__(self, num_classes=21):
        super(FasterRCNN, self).__init__()
        self.transform = FasterRCNNTransform()
        self.backbone = BackBone()
        out_channels = 256

        # RPN
        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        # 根据尺度生成框
        rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        # 滑动窗口进行预测
        rpn_head = RPNHead(out_channels, rpn_anchor_generator.num_anchors_per_location()[0])
        # 创建rpn部分
        self.rpn = RegionProposalNetwork(rpn_anchor_generator, rpn_head)

        # ROI
        box_roi_pool = MultiScaleRoIAlign(output_size=7, sampling_ratio=2)

        # 两个全连接层
        resolution = box_roi_pool.output_size[0]
        representation_size = 1024
        box_head = TwoMLPHead(out_channels * resolution ** 2, representation_size)
        # 预测层
        box_predictor = FastRCNNPredictor(representation_size, num_classes)

        self.roi_heads = RoIHeads(box_roi_pool, box_head, box_predictor)

    def forward(self, images, targets=None, device='cpu'):
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            original_image_sizes.append((val[0], val[1]))

        # transform 预处理
        images, targets = self.transform(images, targets, device)

        # 计算特征图
        features = self.backbone(images.tensors)
        # print([f.shape for f in features])

        # 计算rpn
        proposals, proposal_losses = self.rpn(images, features, targets)
        # print([p.shape for p in proposals], proposal_losses)

        # 计算 roi
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        # 测试时将 图像 转换成 真实值
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if self.training:
            return losses
        else:
            return detections
