import torch
from torch.utils import data
from torchvision import transforms
from torchvision import datasets

Classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
           'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


def deal_target(key):
    """对 xml 的 dic 进行处理"""
    ann = key['annotation']
    img_h = float(ann['size']['height'])
    img_w = float(ann['size']['width'])
    ob = ann['object']
    boxes = []
    labels = []
    for o in ob:
        cls = o['name']
        # x1 = float(o['bndbox']['xmin']) / img_w
        # y1 = float(o['bndbox']['ymin']) / img_h
        # x2 = float(o['bndbox']['xmax']) / img_w
        # y2 = float(o['bndbox']['ymax']) / img_h
        x1 = float(o['bndbox']['xmin'])
        y1 = float(o['bndbox']['ymin'])
        x2 = float(o['bndbox']['xmax'])
        y2 = float(o['bndbox']['ymax'])
        boxes.append([x1, y1, x2, y2])
        labels.append(Classes.index(cls))

    target = {'boxes': torch.as_tensor(boxes, dtype=torch.float32),
              'labels': torch.as_tensor(labels, dtype=torch.int64)}
    return target


def collate_fn(batch):
    return tuple(zip(*batch))


def get_voc_iter(path, batch_size):
    trans = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.VOCDetection(path, year='2012', image_set='train', download=False, transform=trans,
                                      target_transform=deal_target)
    test_set = datasets.VOCDetection(path, year='2007', image_set='test', download=False, transform=trans,
                                     target_transform=deal_target)
    return data.DataLoader(train_set, batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn), \
           data.DataLoader(test_set, batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn)
