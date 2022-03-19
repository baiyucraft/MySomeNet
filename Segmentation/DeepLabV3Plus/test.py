import os

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

import network
import utils
from train import config


def test():
    trans = transforms.Compose([transforms.Resize(2000),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225]), ])
    model = network.modeling.deeplabv3plus_resnet50(num_classes=2)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    if config['ckpt'] is not None and os.path.isfile(config['ckpt']):
        checkpoint = torch.load(config['ckpt'], map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model.to(config['device'])

    model = model.eval()

    root = 'img'
    for img in os.listdir(root):
        img_path = os.path.join(root, img)
        img_o = trans(Image.open(img_path).convert('RGB'))
        img = img_o.unsqueeze(0).to(config['device'])
        pred = model(img).argmax(axis=1).cpu()
        pred = pred.squeeze().permute(2, 0, 1).float()

        _, axes = plt.subplots(1, 2)
        axes[0].imshow(img_o.cpu().squeeze().permute(1, 2, 0))
        axes[0].get_xaxis().set_visible(False)
        axes[0].get_yaxis().set_visible(False)
        axes[1].imshow(pred.squeeze())
        axes[1].get_xaxis().set_visible(False)
        axes[1].get_yaxis().set_visible(False)
        plt.show()
