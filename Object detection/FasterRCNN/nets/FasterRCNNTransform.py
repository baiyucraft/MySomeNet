import math
import torch
from torch import nn
from torch.nn import functional


def _resize_image(image, self_min_size, self_max_size, target):
    im_shape = torch.tensor(image.shape[-2:])
    min_size = float(torch.min(im_shape))
    max_size = float(torch.max(im_shape))

    scale_factor = self_min_size / min_size
    if max_size * scale_factor > self_max_size:
        scale_factor = self_max_size / max_size

    image = functional.interpolate(image[None], scale_factor=scale_factor, mode='bilinear', align_corners=False,
                                   recompute_scale_factor=True)[0]

    if target is None:
        return image, target

    return image, target


def _resize_boxes(boxes, original_size, new_size):
    ratios = [
        torch.tensor(s, dtype=torch.float32, device=boxes.device) /
        torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    ratio_height, ratio_width = ratios
    xmin, ymin, xmax, ymax = boxes.unbind(1)

    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)


class ImageList(object):

    def __init__(self, tensors, image_sizes):
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        return ImageList(cast_tensor, self.image_sizes)


class FasterRCNNTransform(nn.Module):
    """targets: {boxes = [x1,y1,x2,y2], labels}"""

    def __init__(self, min_size=800, max_size=1333):
        super(FasterRCNNTransform, self).__init__()
        self.min_size = min_size
        self.max_size = max_size
        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]

    def forward(self, images, targets, device):
        images = [img.to(device) for img in images]
        if targets is not None:
            targets_copy = []
            for t in targets:
                data = {}
                for k, v in t.items():
                    if isinstance(v, torch.Tensor):
                        data[k] = v.to(device)
                    else:
                        data[k] = v
                targets_copy.append(data)
            targets = targets_copy

        for i in range(len(images)):
            image = images[i]
            target_index = targets[i] if targets is not None else None

            # 标准化和改变大小
            image = self.normalize(image)
            image, target_index = self.resize(image, target_index)

            images[i] = image
            if targets is not None:
                targets[i] = target_index

        # 存储图片大小
        image_sizes = [img.shape[-2:] for img in images]

        # 返回一个批次的图片用于训练
        images = self.batch_images(images)
        image_sizes_list = []
        for image_size in image_sizes:
            image_sizes_list.append((image_size[0], image_size[1]))

        image_list = ImageList(images, image_sizes_list)
        return image_list, targets

    def normalize(self, image):
        """标准化"""
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        return (image - mean[:, None, None]) / std[:, None, None]

    def resize(self, image, target):
        """改变大小"""
        h, w = image.shape[-2:]
        image, target = _resize_image(image, float(self.min_size), float(self.max_size), target)

        if target is None:
            return image, target

        bbox = target["boxes"]
        bbox = _resize_boxes(bbox, (h, w), image.shape[-2:])
        target["boxes"] = bbox

        return image, target

    def batch_images(self, images, size_divisible=32):
        """返回一个批次的图片，图片大小为最大值"""
        the_list = [list(img.shape) for img in images]
        max_size = the_list[0]
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                max_size[index] = max(max_size[index], item)

        stride = float(size_divisible)
        max_size = list(max_size)
        max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
        max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)

        batch_shape = [len(images)] + max_size
        batched_imgs = images[0].new_full(batch_shape, 0)
        for img, pad_img in zip(images, batched_imgs):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

        return batched_imgs

    def postprocess(self, result, image_shapes, original_image_sizes):
        """在测试时将图像转换为原来的图像"""
        if self.training:
            return result
        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
            boxes = pred["boxes"]
            boxes = _resize_boxes(boxes, im_s, o_im_s)
            result[i]["boxes"] = boxes
        return result
