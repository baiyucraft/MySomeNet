import torch
from torch import optim
from nets.YOLOv3 import YOLOv3
from utils.config import Config
from utils.data import get_voc_iter
from utils.train_utils import weights_init, train, train_ones
from utils.show_utils import try_gpu


def get_model_weights(net):
    path = './model_data/yolov3.pt'
    pt_model = torch.load(path)['model'].model
    pt_param = list(pt_model.parameters())

    net_dic = net.state_dict()
    net_param = net.named_parameters()
    up_dic = {}
    for i, (name, parameters) in enumerate(net_param):
        if i < 156:
            up_dic[name] = pt_param[i].data

    net_dic.update(up_dic)
    net.load_state_dict(net_dic)
    print('update true')

    for i, p in enumerate(net.parameters()):
        if i < 156:
            p.requires_grad = False
    print('freeze true')


def freeze(net):
    for i, p in enumerate(net.parameters()):
        if i < 156:
            p.requires_grad = False
    print('freeze true')


if __name__ == '__main__':
    num_epoch, batch_size, lr, = 100, 4, 5e-4
    device = try_gpu()
    # device = 'cpu'
    net = YOLOv3()

    net_param_path = 'model_data/net.params'
    log_path = 'logs/logs.txt'
    weights_init(net, path=net_param_path)
    # get_model_weights(net)
    freeze(net)

    train_iter, test_iter = get_voc_iter('../../dataset', batch_size, size=Config['Size'])

    updater = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
    updater_scheduler = optim.lr_scheduler.ReduceLROnPlateau(updater, mode='min', factor=0.5, patience=2, verbose=True)

    # train_ones(net, train_iter, device, mode='train')
    train(net, train_iter, num_epoch, updater, updater_scheduler, device, net_param_path, log_path,
          train_threshold=1.524)
