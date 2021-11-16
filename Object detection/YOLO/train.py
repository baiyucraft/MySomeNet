import torch
from torch import optim

from nets.YOLOv1 import YOLOv1
from utils.data import get_voc_iter
from utils.train_utils import weights_init, train, train_ones
from utils.show_utils import try_gpu

if __name__ == '__main__':
    num_epoch, batch_size, lr, = 100, 1, 5e-4
    # device = try_gpu()
    device = 'cpu'
    net = YOLOv1()

    net_param_path = 'model_data/net.params'
    log_path = 'logs/logs.txt'
    weights_init(net, path=net_param_path)

    train_iter, test_iter = get_voc_iter('../../dataset', batch_size, size=(448, 448))

    updater = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
    updater_scheduler = optim.lr_scheduler.ReduceLROnPlateau(updater, mode='min', factor=0.5, patience=2, verbose=True)

    # train_ones(net, train_iter, device, mode='test')
    train(net, train_iter, num_epoch, updater, updater_scheduler, device, net_param_path, log_path)
