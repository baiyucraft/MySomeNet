import os
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm


def weights_init(net, init_type='normal', init_gain=0.01, path=None):
    """权重初始化"""

    def init_func(m):
        # 卷积层权重
        if m.__class__.__name__ == 'Conv2d':
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
        # 归一层
        elif m.__class__.__name__ == 'BatchNorm2d':
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    if path and os.path.exists(path):
        print(f'initialize network with path')
        net.load_state_dict(torch.load(path))
    else:
        print(f'initialize network with {init_type} type')
        net.apply(init_func)


def update_log(root, loc_loss, conf_loss):
    with open(root, 'r', encoding='utf-8') as f:
        loc_loss_list = [float(ll) for ll in f.readline().split()]
        conf_loss_list = [float(cl) for cl in f.readline().split()]

    loc_loss_list.append(loc_loss)
    conf_loss_list.append(conf_loss)

    with open(root, 'w', encoding='utf-8') as f:
        f.writelines(' '.join([f'{ll:.3f}' for ll in loc_loss_list]) + '\n')
        f.writelines(' '.join([f'{cl:.3f}' for cl in conf_loss_list]))


def train_epoch(net, train_iter, loss, updater, device, mode='train'):
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


def train(net, train_iter, num_epoch, loss, updater, updater_scheduler, device, net_param_path, log_path,
          train_threshold=None):
    net.to(device)

    for epoch in range(num_epoch):
        # train
        train_loc_loss, train_conf_loss = train_epoch(net, train_iter, loss, updater, device, mode='train')
        print(f'{epoch + 1}, {train_loc_loss:.3f}, {train_conf_loss:.3f}')
        updater_scheduler.step(train_loc_loss + train_conf_loss)

        # eval
        # val_loc_loss, val_conf_loss = train_epoch(net, train_iter, updater, device, mode='eval')
        # print(f'{epoch + 1}, {val_loc_loss:.3f}, {val_conf_loss:.3f}')

        if train_threshold is None:
            torch.save(net.state_dict(), net_param_path)
            update_log(log_path, train_loc_loss, train_conf_loss)
            print('save model param and log')
            train_threshold = train_conf_loss
        elif train_conf_loss < train_threshold:
            torch.save(net.state_dict(), net_param_path)
            update_log(log_path, train_loc_loss, train_conf_loss)
            print('save model param and log')
            train_threshold = train_conf_loss


def set_figsize(figsize=(3.5, 2.5)):
    """设置matplotlib的图表大小。"""
    plt.rcParams['figure.figsize'] = figsize


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置matplotlib的轴。"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None, tpye=0):
    """绘制数据点。"""
    if legend is None:
        legend = []

    # 设置画布大小
    set_figsize(figsize)
    # 移动坐标轴
    axes = axes if axes else plt.gca()

    # 如果 `X` 有一个轴，输出True
    def has_one_axis(X):
        return hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list) and not hasattr(X[0], "__len__")

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    # 清除当前活动轴
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
    if not tpye:
        plt.show()


def plot_log(size=300):
    with open(f'../logs/m_mask_ssd{size}.log', 'r', encoding='utf-8') as f:
        loc_loss_list = [float(ll) for ll in f.readline().split()]
        conf_loss_list = [float(cl) for cl in f.readline().split()]
    num_epoch = len(loc_loss_list)
    print(num_epoch)
    plot(X=[range(1, num_epoch + 1)], Y=[loc_loss_list, conf_loss_list], xlabel='epoch', ylabel='loss',
         xlim=[1, num_epoch], legend=['loc_loss', 'conf_loss'], figsize=(6, 4))


if __name__ == '__main__':
    plot_log(300)
    # plot_log(512)
