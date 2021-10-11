import os.path
import warnings
import json
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from nets.ssd import get_ssd
from nets.ssd_training import MultiBoxLoss, weights_init, get_vgg_pth, get_already_pth
from utils.config import Config

from utils.data import get_voc_iter
from utils.utils import try_gpu


def tes_xy():
    x = torch.randn(size=(2, 3, 300, 300))
    targets = torch.Tensor([[[0.1, 0.1, 0.2, 0.2, 2], [0.1, 0.1, 0.2, 0.2, 2]],
                            [[0.2, 0.2, 0.3, 0.3, 5], [0.2, 0.2, 0.3, 0.3, 5]]])
    return x, targets


def train_epoch(net, train_iter, updater, device, mode='train'):
    loc_loss = 0
    conf_loss = 0
    count = 0
    for X, y in tqdm(train_iter):
        if mode == 'train':
            net.train()
        else:
            net.eval()
        X, y = X.to(device), y.to(device)
        output = net(X)
        loss_l, loss_c = loss(output, y)
        all_loss = loss_l + loss_c

        # 反向传播
        if mode == 'train':
            updater.zero_grad()
            all_loss.backward()
            updater.step()

        loc_loss += loss_l.item()
        conf_loss += loss_c.item()
        count += 1

    return loc_loss / count, conf_loss / count


def train(net, train_iter, updater, updater_scheduler, device):
    net.to(device)
    for epoch in range(num_epoch):
        # train
        train_loc_loss, train_conf_loss = train_epoch(net, train_iter, updater, device, mode='train')
        print(f'{epoch + 1}, {train_loc_loss:.3f}, {train_conf_loss:.3f}')
        updater_scheduler.step(train_loc_loss + train_conf_loss)

        # eval
        # val_loc_loss, val_conf_loss = train_epoch(net, train_iter, updater, device, mode='eval')
        # print(f'{epoch + 1}, {val_loc_loss:.3f}, {val_conf_loss:.3f}')

        # updater_scheduler.step(val_loc_loss + val_conf_loss)
        torch.save(net.state_dict(), net_param_path)


if __name__ == "__main__":
    num_epoch, batch_size, lr, = 100, 6, 5e-6
    device = try_gpu()
    # 主干网络权重
    bone_net_path = "model_data/vgg16-397923af.pth"
    net_param_path = 'model_data/net.params'
    ssd_weight_pth = 'model_data/ssd_weights.pth'

    # 加载模型
    net = get_ssd("train", Config["num_classes"])
    # 自己训练的
    weights_init(net, path=net_param_path)
    # get_vgg_pth(net, bone_net_path)
    # 原来训练好的
    # get_already_pth(net, ssd_weight_pth)

    loss = MultiBoxLoss(Config['num_classes'], 0.5, True, device=device)
    updater = optim.Adam(net.parameters(), lr=lr)
    updater_scheduler = optim.lr_scheduler.ReduceLROnPlateau(updater, mode='min', factor=0.5, patience=2, verbose=True)
    train_iter, test_iter = get_voc_iter('../dataset', batch_size, [300, 300])

    train(net, train_iter, updater, updater_scheduler, device)
