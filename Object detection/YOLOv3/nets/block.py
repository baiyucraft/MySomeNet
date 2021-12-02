import torch
from torch import nn
from torch.nn import functional as F
from utils.box_utils import b_box_to_c_box, c_box_to_b_box, box_iou, clip_boxes_to_image, get_anchors_iou, bbox_iou
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


def smooth_BCE(eps=0.1):
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class YOLOv3Loss(nn.Module):
    def __init__(self):
        super(YOLOv3Loss, self).__init__()
        self.size = Config['Size'][0]
        self.num_classes = len(Config['Classes'])
        self.anchors = torch.Tensor(Config['Anchors']).reshape(3, 3, 2) * self.size
        self.nl = self.anchors.shape[0]

        self.anchor_t = 4.0  # anchor-multiple threshold
        self.gr = 1.0
        self.cp, self.cn = smooth_BCE()  # positive, negative BCE targets# Define criteria
        self.BCE_cls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]))
        self.BCE_obj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]))
        self.balance = [4.0, 1.0, 0.4]
        self.box_loss = 0.05
        self.cls_loss = 0.5
        self.obj_loss = 1.0

    def forward(self, pred, targets, device):
        # pred(box,obj,cls) target(b,cls,box,a)
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self.build_t(pred, targets, device)
        # Losses
        for i, pi in enumerate(pred):
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio

                # Classification
                if self.num_classes > 1:
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCE_cls(ps[:, 5:], t)  # BCE

            obji = self.BCE_obj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss

        lbox *= self.box_loss
        lobj *= self.obj_loss
        lcls *= self.cls_loss
        bs = tobj.shape[0]

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_t(self, pred, targets, device):
        c1, c2, c3 = pred
        batch_size, num_anchors, h, w, b = c1.shape
        nt = targets.shape[0]
        tcls, tbox, indices, anch = [], [], [], []
        targets[:, 2:] = b_box_to_c_box(targets[:, 2:])

        gain = torch.ones(7, device=device)  # normalized to gridspace gain
        ai = torch.arange(num_anchors, device=device).float().reshape(num_anchors, 1)
        ai = ai.repeat(1, nt)
        targets = torch.cat((targets.repeat(num_anchors, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], ], device=device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i].to(device)
            gain[2:6] = torch.tensor(pred[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            # wh ratio
            r = t[:, :, 4:6] / anchors[:, None]
            j = torch.max(r, 1 / r).max(2)[0] < self.anchor_t  # compare
            t = t[j]  # filter

            # Offsets
            gxy = t[:, 2:4]  # grid xy
            gxi = gain[[2, 3]] - gxy  # inverse
            j, k = ((gxy % 1 < g) & (gxy > 1)).T
            l, m = ((gxi % 1 < g) & (gxi > 1)).T
            j = torch.stack((torch.ones_like(j), j, k, l, m))
            t = t.repeat((5, 1, 1))[j]
            offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class
        return tcls, tbox, indices, anch


class YOLOPredict(nn.Module):
    def __init__(self, confidence=0.1, iou_threshold=0.5):
        super(YOLOPredict, self).__init__()
        self.num_classes = len(Config['Classes'])
        self.confidence = confidence
        self.iou_threshold = iou_threshold

    def forward(self, pred, device):
        box_pred = pred[..., :4].reshape(-1, 4)
        obj_pred = pred[..., 4].reshape(-1, 1)
        score_pred = pred[..., 5:].reshape(-1, self.num_classes)

        c_boxes = c_box_to_b_box(box_pred)
        # 将框限制在 0-1
        c_boxes = clip_boxes_to_image(c_boxes, (1, 1))
        scores = score_pred * obj_pred

        # 针对每个图片的每个框 计算 nms
        output = []
        p_boxes = []
        p_labels = []
        p_scores = []

        for cl in range(self.num_classes):
            # 满足最低阈值
            cl_idx = scores[:, cl] > self.confidence
            if cl_idx.sum() == 0:
                continue
            # nms
            cl_box = c_boxes[cl_idx]
            cl_scores = scores[cl_idx, cl]
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
