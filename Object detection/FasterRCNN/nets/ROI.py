import torch
from torch import nn, Tensor
from torch.nn import functional as F
from utils.box_utils import box_iou, Matcher, BalancedPositiveNegativeSampler, BoxCoder, box_area, smooth_l1_loss, \
    clip_boxes_to_image, remove_small_boxes, batched_nms


class LevelMapper(object):
    """计算box在哪个lvl上"""

    def __init__(self, k_min, k_max, canonical_scale=224, canonical_level=4, eps=1e-6):
        self.k_min = k_min
        self.k_max = k_max
        self.s0 = canonical_scale
        self.lvl0 = canonical_level
        self.eps = eps

    def __call__(self, boxlists):
        """
        对框在哪个特征图上进行pool
        """
        # Compute level ids 相较于方形的面积
        s = torch.sqrt(torch.cat([box_area(boxlist) for boxlist in boxlists]))

        # 公式 in FPN paper
        target_lvls = torch.floor(self.lvl0 + torch.log2(s / self.s0) + torch.tensor(self.eps, dtype=s.dtype))
        target_lvls = torch.clamp(target_lvls, min=self.k_min, max=self.k_max)
        return (target_lvls.to(torch.int64) - self.k_min).to(torch.int64)


class MultiScaleRoIAlign(nn.Module):
    """
    多尺度 ROI Align 池化 FPN中的，``224`` and ``k0=4``

    Args:
        output_size: 输出的大小
        sampling_ratio: roi_align 将用到的参数
    """

    def __init__(self, output_size, sampling_ratio):
        super(MultiScaleRoIAlign, self).__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size)

        self.sampling_ratio = sampling_ratio
        self.output_size = output_size
        self.scales = None
        self.map_levels = None

    def roi_align(self, input, boxes, output_size, spatial_scale=1.0, sampling_ratio=-1, aligned=False):
        """
        Performs Region of Interest (RoI) Align operator described in Mask R-CNN

        Args:
            input (Tensor[N, C, H, W]): input tensor
            boxes (Tensor[K, 5] or List[Tensor[L, 4]]): the box coordinates in (x1, y1, x2, y2)
                format where the regions will be taken from.
                The coordinate must satisfy ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
                If a single Tensor is passed,
                then the first column should contain the batch index. If a list of Tensors
                is passed, then each Tensor will correspond to the boxes for an element i
                in a batch
            output_size (int or Tuple[int, int]): the size of the output after the cropping
                is performed, as (height, width)
            spatial_scale (float): a scaling factor that maps the input coordinates to
                the box coordinates. Default: 1.0
            sampling_ratio (int): number of sampling points in the interpolation grid
                used to compute the output value of each pooled output bin. If > 0,
                then exactly sampling_ratio x sampling_ratio grid points are used. If
                <= 0, then an adaptive number of grid points are used (computed as
                ceil(roi_width / pooled_w), and likewise for height). Default: -1
            aligned (bool): If False, use the legacy implementation.
                If True, pixel shift it by -0.5 for align more perfectly about two neighboring pixel indices.
                This version in Detectron2

        Returns:
            output (Tensor[K, C, output_size[0], output_size[1]])
        """
        return torch.ops.torchvision.roi_align(input, boxes, spatial_scale, output_size[0], output_size[1],
                                               sampling_ratio,
                                               aligned)

    def convert_to_roi_format(self, boxes):
        """将一个 batch 中的所有 box 合并，并加一个参数ids"""
        concat_boxes = torch.cat(boxes, dim=0)
        device, dtype = concat_boxes.device, concat_boxes.dtype
        ids = torch.cat([torch.full_like(b[:, :1], i, dtype=dtype, layout=torch.strided, device=device)
                         for i, b in enumerate(boxes)], dim=0, )
        # 将一个 batch 中的所有 box 合并，并加一个参数ids
        rois = torch.cat([ids, concat_boxes], dim=1)
        return rois

    def infer_scale(self, feature: Tensor, original_size):
        # 推断feature_map_size是图片original_size的2**k分之一，backbone的下采样率为2!
        size = feature.shape[-2:]
        possible_scales = []
        for s1, s2 in zip(size, original_size):
            approx_scale = float(s1) / float(s2)
            scale = 2 ** float(torch.tensor(approx_scale).log2().round())
            possible_scales.append(scale)
        assert possible_scales[0] == possible_scales[1]
        return possible_scales[0]

    def setup_scales(self, features, image_shapes):
        """获得尺度"""
        max_x = 0
        max_y = 0
        for shape in image_shapes:
            max_x = max(shape[0], max_x)
            max_y = max(shape[1], max_y)
        original_input_shape = (max_x, max_y)

        # 下采样的尺度
        scales = [self.infer_scale(feat, original_input_shape) for feat in features]
        # 得到下采样的最小和最大倍数（2 6）
        lvl_min = -torch.log2(torch.tensor(scales[0], dtype=torch.float32)).item()
        lvl_max = -torch.log2(torch.tensor(scales[-1], dtype=torch.float32)).item()
        self.scales = scales
        self.map_levels = LevelMapper(k_min=int(lvl_min), k_max=int(lvl_max))

    def forward(self, x, boxes, image_shapes):
        """
        Args:
            x: 特征图
            boxes: rpn层输出的 proposals
            image_shapes: 图片的原大小
        """
        x_filtered = x[:-1]
        num_levels = len(x_filtered)
        # shape(-1, 5)
        rois = self.convert_to_roi_format(boxes)
        self.setup_scales(x_filtered, image_shapes)

        scales = self.scales

        if num_levels == 1:
            return self.roi_align(x_filtered[0], rois, output_size=self.output_size, spatial_scale=scales[0],
                                  sampling_ratio=self.sampling_ratio)

        mapper = self.map_levels
        # 每个框对应的lvl
        levels = mapper(boxes)
        num_rois = len(rois)
        num_channels = x_filtered[0].shape[1]

        dtype, device = x_filtered[0].dtype, x_filtered[0].device
        # (num_boxes, num_channels, out_size, out_size)
        result = torch.zeros((num_rois, num_channels,) + self.output_size, dtype=dtype, device=device, )

        for level, (per_level_feature, scale) in enumerate(zip(x_filtered, scales)):
            idx_in_level = torch.where(levels == level)[0]
            rois_per_level = rois[idx_in_level]

            # roi_align
            result_idx_in_level = self.roi_align(per_level_feature, rois_per_level, self.output_size,
                                                 spatial_scale=scale, sampling_ratio=self.sampling_ratio)

            result[idx_in_level] = result_idx_in_level.to(result.dtype)

        return result


class TwoMLPHead(nn.Module):
    """两个全连接层"""

    def __init__(self, in_channels, representation_size):
        super(TwoMLPHead, self).__init__()
        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.relu(self.fc6(x))
        x = self.relu(self.fc7(x))
        return x


class FastRCNNPredictor(nn.Module):
    """框的类别预测和回归预测"""

    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    """计算损失
    Args:
        class_logits: 类别预测值
        box_regression: 偏移回归预测值
        labels: 类别真实值
        regression_targets: 回归真实值
    """

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    classification_loss = F.cross_entropy(class_logits, labels)

    # 大于0的为目标, 对目标计算损失
    sampled_pos_inds_subset = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, num_classes, 4)

    box_loss = smooth_l1_loss(box_regression[sampled_pos_inds_subset, labels_pos],
                              regression_targets[sampled_pos_inds_subset],
                              size_average=False, ) / labels.numel()

    return classification_loss, box_loss


class RoIHeads(nn.Module):

    def __init__(self, box_roi_pool, box_head, box_predictor,
                 # Faster R-CNN training
                 high_threshold=0.5, low_threshold=0.5,
                 batch_size_per_image=512, positive_fraction=0.25,
                 # Faster R-CNN inference
                 score_thresh=0.05,
                 nms_thresh=0.5,
                 detections_per_img=100 ):
        super(RoIHeads, self).__init__()

        self.box_similarity = box_iou
        # assign ground-truth boxes for each proposal
        self.proposal_matcher = Matcher(high_threshold, low_threshold)

        self.fg_bg_sampler = BalancedPositiveNegativeSampler(batch_size_per_image, positive_fraction)

        self.box_coder = BoxCoder((10., 10., 5., 5.))

        self.box_roi_pool = box_roi_pool
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        """将目标与框对应"""
        matched_idxs = []
        labels = []
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):
            match_quality_matrix = self.box_similarity(gt_boxes_in_image, proposals_in_image)
            # 得到每个 proposals 对应的值
            matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)

            clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)
            # labels_in_image 为所有被认为是目标的 proposals 对应的最大iou的 gt_labels（放入labels_in_image）?类别编号
            labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
            labels_in_image = labels_in_image.to(dtype=torch.int64)

            # 背景设为0
            bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
            labels_in_image[bg_inds] = 0

            # 忽略的设为-1
            ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
            labels_in_image[ignore_inds] = -1

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
        return matched_idxs, labels

    def subsample(self, labels):
        # 平衡正负样本数
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)

        sampled_inds = []
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(zip(sampled_pos_inds, sampled_neg_inds)):
            img_sampled_inds = torch.where(pos_inds_img | neg_inds_img)[0]
            sampled_inds.append(img_sampled_inds)
        return sampled_inds

    def select_training_samples(self, proposals, targets):
        dtype = proposals[0].dtype
        device = proposals[0].device

        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]

        # 将 gt 放入 proposals
        proposals = [torch.cat((proposal, gt_box)) for proposal, gt_box in zip(proposals, gt_boxes)]

        # 将 gt_labels 和 proposals 对应
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
        # 选出一定数量的样本
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        num_images = len(proposals)
        for img_id in range(num_images):
            # 取出对应样本
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

            # 将样本对应的真实框取出
            gt_boxes_in_image = gt_boxes[img_id]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])

        # 计算回归值
        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, matched_idxs, labels, regression_targets

    def postprocess_detections(self, class_logits, box_regression, proposals, image_shapes):
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)
        # 对每个框的类别作softmax
        pred_scores = F.softmax(class_logits, -1)

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
            boxes = clip_boxes_to_image(boxes, image_shape)

            # 为每个预测框创建labels
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # 删除背景，第0列是背景
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # 每个框（batch_size*anchor_num）
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # 删除低分的预测框
            inds = torch.where(scores > self.score_thresh)[0]
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # 移除小框
            keep = remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # 非极大抑制 对不同类的框进行nms
            keep = batched_nms(boxes, scores, labels, self.nms_thresh)
            # 取前 top 的框
            keep = keep[:self.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels

    def forward(self, features, proposals, image_shapes, targets=None):
        """
        Args:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """

        if self.training:
            # training时，进行正负平衡采样，以及为proposal匹配gt，届时计算proposal与匹配的gt之间的loss。
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        # (batch_size * num_boxes, num_channels, out_size, out_size)
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        result = []
        losses = {}
        if self.training:
            loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
            losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append({"boxes": boxes[i], "labels": labels[i], "scores": scores[i]})

        return result, losses
