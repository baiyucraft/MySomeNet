import os

from utils import ext_transforms as et
from PIL import Image
from torch.utils import data
from torchvision import transforms


class MyDataSet(data.Dataset):
    def __init__(self, root='datasets/weeds'):
        img_dir = os.path.join(root, 'images')
        mask_dir = os.path.join(root, 'masks')
        self.num_classes = 2

        self.trans = et.ExtCompose([
            # et.ExtResize(size=opts.crop_size),
            et.ExtRandomScale((0.5, 2.0)),
            et.ExtRandomCrop(size=(513, 513), pad_if_needed=True),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        self.images = [os.path.join(img_dir, d) for d in os.listdir(img_dir)]
        self.masks = [os.path.join(mask_dir, d) for d in os.listdir(mask_dir)]

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])
        return self.trans(img, target)

    def __len__(self):
        return len(self.images)
