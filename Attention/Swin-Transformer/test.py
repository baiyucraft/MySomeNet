import torch
from torchvision import transforms
from PIL import Image

from main import parse_option
from models import build_model
from utils import load_checkpoint

if __name__ == '__main__':
    _, config = parse_option()

    model = build_model(config)
    model.cuda()
    checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)

    img = Image.open('img/cat2.jpg')

    # x = torch.randn((2, 3, 224, 224)).cuda()
    x = transforms.Compose((transforms.Resize(224),
                            transforms.ToTensor(),
                            ))(img).unsqueeze(0).cuda()
    y = model(x)

    print(y.argmax(-1))
