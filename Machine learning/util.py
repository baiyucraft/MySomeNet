import os
import time
import numpy as np
import torch
from IPython import display
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm


def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()。"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


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


class Animator:
    """在动画中绘制数据。"""

    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: set_axes(self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        # display.display(self.fig)
        # display.clear_output(wait=True)


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """展示一系列图片"""
    figsize = (num_cols * scale, num_rows * scale)
    # num_rows行，num_cols列的子图
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    # flatten()使axes方便迭代
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        ax.imshow(img)
        # 不显示x轴与y轴
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])


class Timer:
    """记录多次运行时间。"""

    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器。"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中。"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间。"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和。"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间。"""
        return np.array(self.times).cumsum().tolist()


def weights_init(net, path=None):
    if path and os.path.exists(path):
        print(f'initialize network with path')
        net.load_state_dict(torch.load(path))


def train(net, train_iter, num_epochs, learning_rate, path):
    device = try_gpu()
    net = net.to(device)
    loss = nn.CrossEntropyLoss()
    trainer = torch.optim.SGD(net.parameters(), learning_rate, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(trainer, T_max=100)

    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], legend=['net_loss', 'net_acc'])
    timer = Timer()
    for epoch in tqdm(range(num_epochs)):
        timer.start()
        loss_l, acc_l = [], []
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            X = X.reshape(X.shape[0], -1)
            y_hat = net(X)
            y_loss = loss(y_hat, y)

            trainer.zero_grad()
            y_loss.backward()
            trainer.step()

            y_acc = (y_hat.argmax(axis=1) == y).float().sum() / y.shape[0]
            loss_l.append(y_loss.item())
            acc_l.append(y_acc.item())

        net_loss, net_acc = sum(loss_l) / len(loss_l), sum(acc_l) / len(acc_l)
        animator.add(epoch + 1, (net_loss, net_acc))
        timer.stop()
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f'[ {epoch + 1} ], net loss: {net_loss:.3f}, net acc: {net_acc:.3f}')
            torch.save(net.state_dict(), path)
    print(f'all use: {timer.sum() / 60 :.3f}, '
          f'net loss: {net_loss:.3f}, net acc: {net_acc:.3f}')
    plt.show()

