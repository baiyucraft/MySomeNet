import torch
import torch.nn as nn
from nets.all_blocks import *

if __name__ == '__main__':
    x = torch.randn(size=(1, 3, 300, 300))
    # get_layer(ssd, x)
