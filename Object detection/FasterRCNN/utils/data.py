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


def deal_target(key, p=10):
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
    while len(boxes) < p:
        boxes *= 2
    return torch.Tensor(boxes[:p])


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
