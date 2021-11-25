import torch
from torch.utils import data
from torchvision import transforms
from torchvision import datasets

from utils.config import Config


def deal_target(key):
    """对 xml 的 dic 进行处理"""
    ann = key['annotation']
    img_h = float(ann['size']['height'])
    img_w = float(ann['size']['width'])
    ob = ann['object']
    boxes = []
    # labels = []
    for o in ob:
        cls = o['name']
        x1 = float(o['bndbox']['xmin']) / img_w
        y1 = float(o['bndbox']['ymin']) / img_h
        x2 = float(o['bndbox']['xmax']) / img_w
        y2 = float(o['bndbox']['ymax']) / img_h
        boxes.append([0, Config['Classes'].index(cls), x1, y1, x2, y2])

        # boxes.append([x1, y1, x2, y2])
        # labels.append(Config['Classes'].index(cls))

    # target = {'boxes': torch.as_tensor(boxes, dtype=torch.float32),
    #           'labels': torch.as_tensor(labels, dtype=torch.int64)}
    return torch.Tensor(boxes)


def collate_fn(batch):
    imgs, targets = list(zip(*batch))
    for i, boxes in enumerate(targets):
        boxes[:, 0] = i
    imgs = torch.stack(imgs, dim=0)
    targets = torch.cat(targets, 0)
    return imgs, targets


def get_voc_iter(path, batch_size, size=(448, 448)):
    trans = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize(size),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    train_set = datasets.VOCDetection(path, year='2012', image_set='train', download=False, transform=trans,
                                      target_transform=deal_target)
    test_set = datasets.VOCDetection(path, year='2007', image_set='test', download=False, transform=trans,
                                     target_transform=deal_target)
    return data.DataLoader(train_set, batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn), \
           data.DataLoader(test_set, batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn)


if __name__ == '__main__':
    ite, _ = get_voc_iter(path='../../../dataset', batch_size=2)
    net = next(iter(ite))
    print(net)
