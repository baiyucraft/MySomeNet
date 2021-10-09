import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils import data
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision import datasets

Classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

"""
class SSDDataset(Dataset):
    def __init__(self, train_lines, image_size, is_train):
        super(SSDDataset, self).__init__()

        self.train_lines = train_lines
        self.train_batches = len(train_lines)
        self.image_size = image_size
        self.is_train = is_train

    def __len__(self):
        return self.train_batches

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
        """
实时数据增强的随机预处理
"""
        line = annotation_line.split()
        image = Image.open(line[0])
        iw, ih = image.size
        h, w = input_shape
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        if not random:
            # resize image
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2

            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)

            # correct boxes
            box_data = np.zeros((len(box), 5))
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]
                box_data = np.zeros((len(box), 5))
                box_data[:len(box)] = box

            return image_data, box_data

        # 调整图片大小
        new_ar = w / h * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        # 放置图片
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h),
                              (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))
        new_image.paste(image, (dx, dy))
        image = new_image

        # 是否翻转图片
        flip = self.rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # 色域变换
        hue = self.rand(-hue, hue)
        sat = self.rand(1, sat) if self.rand() < .5 else 1 / self.rand(1, sat)
        val = self.rand(1, val) if self.rand() < .5 else 1 / self.rand(1, val)
        x = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue * 360
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:, :, 0] > 360, 0] = 360
        x[:, :, 1:][x[:, :, 1:] > 1] = 1
        x[x < 0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255

        # 调整目标框坐标
        box_data = np.zeros((len(box), 5))
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip:
                box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]  # 保留有效框
            box_data = np.zeros((len(box), 5))
            box_data[:len(box)] = box

        return image_data, box_data

    def __getitem__(self, index):
        lines = self.train_lines

        if self.is_train:
            img, y = self.get_random_data(lines[index], self.image_size[0:2])
        else:
            img, y = self.get_random_data(lines[index], self.image_size[0:2], random=False)

        boxes = np.array(y[:, :4], dtype=np.float32)
        boxes[:, 0] = boxes[:, 0] / self.image_size[1]
        boxes[:, 1] = boxes[:, 1] / self.image_size[0]
        boxes[:, 2] = boxes[:, 2] / self.image_size[1]
        boxes[:, 3] = boxes[:, 3] / self.image_size[0]
        boxes = np.maximum(np.minimum(boxes, 1), 0)

        y = np.concatenate([boxes, y[:, -1:]], axis=-1)

        img = np.array(img, dtype=np.float32)
        tmp_inp = np.transpose(img - MEANS, (2, 0, 1))
        tmp_targets = np.array(y, dtype=np.float32)

        return tmp_inp, tmp_targets


# DataLoader中collate_fn使用
def ssd_dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = np.array(images)
    return images, bboxes
"""


def deal_target(key):
    """对 xml 的 dic 进行处理"""
    ann = key['annotation']
    img_path = ann['filename']
    img_h = float(ann['size']['height'])
    img_w = float(ann['size']['width'])
    ob = ann['object']
    boxes = []
    for o in ob:
        cls = o['name']
        x1 = int(o['bndbox']['xmin']) / img_w
        y1 = int(o['bndbox']['ymin']) / img_h
        x2 = int(o['bndbox']['xmax']) / img_w
        y2 = int(o['bndbox']['ymax']) / img_h
        boxes.append([x1, y1, x2, y2, Classes.index(cls)])
        # boxes.append([x1, y1, x2, y2, Classes.index(cls) + 1])
    # 扩充
    while len(boxes) < 10:
        boxes *= 2
    return torch.Tensor(boxes[:10])


def get_voc_iter(path, batch_size, resize):
    trans = transforms.Compose([transforms.Resize(resize),
                                transforms.ToTensor(),
                                # transforms.ColorJitter(brightness=0.5, contrast=0.5),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    train_set = datasets.VOCDetection(path, year='2012', image_set='train', download=False, transform=trans,
                                      target_transform=deal_target)
    test_set = datasets.VOCDetection(path, year='2007', image_set='test', download=False, transform=trans,
                                     target_transform=deal_target)
    return data.DataLoader(train_set, batch_size, shuffle=True, num_workers=2), \
           data.DataLoader(test_set, batch_size, shuffle=False, num_workers=2)
