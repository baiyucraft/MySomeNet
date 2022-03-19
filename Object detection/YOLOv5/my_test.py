import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml

from models.yolo import Model
from utils.general import print_args, check_yaml
from utils.loss import ComputeLoss
from utils.torch_utils import select_device

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

LOGGER = logging.getLogger(__name__)

hyp = 'data/hyps/hyp.scratch.yaml'
with open(hyp, errors='ignore') as f:
    hyp = yaml.safe_load(f)  # load hyps dict
    if 'anchors' not in hyp:  # anchors commented in hyp.yaml
        hyp['anchors'] = 3


def Convert_ONNX(net, inp):
    net, inp = net.to('cpu'), inp.to('cpu')
    # Export the model
    torch.onnx.export(net,  # model being run
                      inp,  # model input (or a tuple for multiple inputs)
                      "model_data/test.onnx",  # where to save the model
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['modelInput'],  # the model's input names
                      output_names=['modelOutput'],  # the model's output names
                      dynamic_axes={'modelInput': {0: 'batch_size'},  # variable length axes
                                    'modelOutput': {0: 'batch_size'}})
    print(" ")
    print('Model has been converted to ONNX')


if __name__ == '__main__':
    file_name = 'pest_s.yaml'
    file_name = check_yaml(file_name)  # check YAML
    print(file_name)
    device = select_device('')

    model = Model(file_name).to(device)
    model.hyp = hyp
    compute_loss = ComputeLoss(model)
    model.train()
    # model.eval()

    img = torch.rand(1, 3, 640, 640).to(device)
    targets = torch.Tensor([[0, 1, 0.124, 0.572, 0.132, 0.159]]).to(device)

    # train
    pred = model(img)
    loss, loss_items = compute_loss(pred, targets)

    # eval
    # y = model(img)
    # for i in y:
    #     print(i.shape)

    # Convert_ONNX(model, img)
