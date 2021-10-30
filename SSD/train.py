import torch.optim as optim

from nets.loss import MultiBoxLoss
from nets.ssd import get_ssd
from utils.config import Config

from utils.data import get_voc_iter
from utils.train_ssd import weights_init, train
from utils.utils import try_gpu

if __name__ == "__main__":
    num_epoch, batch_size, lr, = 100, 6, 5e-4
    device = try_gpu()
    # 主干网络权重
    bone_net_path = "model_data/vgg16-397923af.pth"
    net_param_path = 'model_data/net.params'
    ssd_weight_pth = 'model_data/ssd_weights.pth'
    log_path = 'logs/log.txt'

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

    train(net, train_iter, num_epoch, loss, updater, updater_scheduler, device, net_param_path, log_path)
