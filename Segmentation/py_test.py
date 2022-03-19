import os

import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image
from torchvision.models import segmentation

from voc_data import VOC_COLORMAP


def label2image(pred):
    colormap = torch.tensor(VOC_COLORMAP)
    X = pred.long()
    return colormap[X, :]


if __name__ == '__main__':
    trans = transforms.Compose((transforms.Resize(320),
                                transforms.ToTensor()))

    model = segmentation.deeplabv3_resnet101(True)
    # model = segmentation.deeplabv3_mobilenet_v3_large(True)

    for p in os.listdir('DeepLabV3Plus/img'):
        img = Image.open('DeepLabV3Plus/img/' + p)
        plt.axis('off')
        plt.imshow(img)
        plt.show()
        img = trans(img).unsqueeze(0)

        models, img = model.cuda().eval(), img.cuda()
        y = model(img)['out'].argmax(axis=1).cpu()
        y_ = label2image(y).squeeze().permute(2, 0, 1).float()
        plt.axis('off')
        plt.imshow(transforms.ToPILImage()(y_))
        plt.show()
    # img =
