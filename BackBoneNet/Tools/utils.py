import os
import time

import numpy as np
import torch
import torchvision
from IPython import display
from matplotlib import pyplot as plt
from torch import nn
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm

class_name = ['ak47', 'american-flag', 'backpack', 'baseball-bat', 'baseball-glove', 'basketball-hoop', 'bat',
              'bathtub', 'bear', 'beer-mug', 'billiards', 'binoculars', 'birdbath', 'blimp', 'bonsai-101', 'boom-box',
              'bowling-ball', 'bowling-pin', 'boxing-glove', 'brain-101', 'breadmaker', 'buddha-101', 'bulldozer',
              'butterfly', 'cactus', 'cake', 'calculator', 'camel', 'cannon', 'canoe', 'car-tire', 'cartman', 'cd',
              'centipede', 'cereal-box', 'chandelier-101', 'chess-board', 'chimp', 'chopsticks', 'cockroach',
              'coffee-mug', 'coffin', 'coin', 'comet', 'computer-keyboard', 'computer-monitor', 'computer-mouse',
              'conch', 'cormorant', 'covered-wagon', 'cowboy-hat', 'crab-101', 'desk-globe', 'diamond-ring', 'dice',
              'dog', 'dolphin-101', 'doorknob', 'drinking-straw', 'duck', 'dumb-bell', 'eiffel-tower',
              'electric-guitar-101', 'elephant-101', 'elk', 'ewer-101', 'eyeglasses', 'fern', 'fighter-jet',
              'fire-extinguisher', 'fire-hydrant', 'fire-truck', 'fireworks', 'flashlight', 'floppy-disk',
              'football-helmet', 'french-horn', 'fried-egg', 'frisbee', 'frog', 'frying-pan', 'galaxy', 'gas-pump',
              'giraffe', 'goat', 'golden-gate-bridge', 'goldfish', 'golf-ball', 'goose', 'gorilla', 'grand-piano-101',
              'grapes', 'grasshopper', 'guitar-pick', 'hamburger', 'hammock', 'harmonica', 'harp', 'harpsichord',
              'hawksbill-101', 'head-phones', 'helicopter-101', 'hibiscus', 'homer-simpson', 'horse', 'horseshoe-crab',
              'hot-air-balloon', 'hot-dog', 'hot-tub', 'hourglass', 'house-fly', 'human-skeleton', 'hummingbird',
              'ibis-101', 'ice-cream-cone', 'iguana', 'ipod', 'iris', 'jesus-christ', 'joy-stick', 'kangaroo-101',
              'kayak', 'ketch-101', 'killer-whale', 'knife', 'ladder', 'laptop-101', 'lathe', 'leopards-101',
              'license-plate', 'lightbulb', 'light-house', 'lightning', 'llama-101', 'mailbox', 'mandolin', 'mars',
              'mattress', 'megaphone', 'menorah-101', 'microscope', 'microwave', 'minaret', 'minotaur',
              'motorbikes-101', 'mountain-bike', 'mushroom', 'mussels', 'necktie', 'octopus', 'ostrich', 'owl',
              'palm-pilot', 'palm-tree', 'paperclip', 'paper-shredder', 'pci-card', 'penguin', 'people',
              'pez-dispenser', 'photocopier', 'picnic-table', 'playing-card', 'porcupine', 'pram', 'praying-mantis',
              'pyramid', 'raccoon', 'radio-telescope', 'rainbow', 'refrigerator', 'revolver-101', 'rifle',
              'rotary-phone', 'roulette-wheel', 'saddle', 'saturn', 'school-bus', 'scorpion-101', 'screwdriver',
              'segway', 'self-propelled-lawn-mower', 'sextant', 'sheet-music', 'skateboard', 'skunk', 'skyscraper',
              'smokestack', 'snail', 'snake', 'sneaker', 'snowmobile', 'soccer-ball', 'socks', 'soda-can', 'spaghetti',
              'speed-boat', 'spider', 'spoon', 'stained-glass', 'starfish-101', 'steering-wheel', 'stirrups',
              'sunflower-101', 'superman', 'sushi', 'swan', 'swiss-army-knife', 'sword', 'syringe', 'tambourine',
              'teapot', 'teddy-bear', 'teepee', 'telephone-box', 'tennis-ball', 'tennis-court', 'tennis-racket',
              'theodolite', 'toaster', 'tomato', 'tombstone', 'top-hat', 'touring-bike', 'tower-pisa', 'traffic-light',
              'treadmill', 'triceratops', 'tricycle', 'trilobite-101', 'tripod', 't-shirt', 'tuning-fork', 'tweezer',
              'umbrella-101', 'unicorn', 'vcr', 'video-projector', 'washing-machine', 'watch-101', 'waterfall',
              'watermelon', 'welding-mask', 'wheelbarrow', 'windmill', 'wine-bottle', 'xylophone', 'yarmulke', 'yo-yo',
              'zebra', 'airplanes-101', 'car-side-101', 'faces-easy-101', 'greyhound', 'tennis-shoes', 'toad',
              'clutter']


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
        display.display(self.fig)
        display.clear_output(wait=True)


def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()。"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def show_image(img, title=None):
    """Plot image"""
    plt.axis('off')
    if torch.is_tensor(img):
        img = img.permute(1, 2, 0)
        plt.imshow(img.numpy())
    else:
        plt.imshow(img)
    if title:
        plt.title(title)
    plt.show()


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5, tpye=0):
    # 图片大小
    figsize = (num_cols * scale, num_rows * scale)
    # num_rows行，num_cols列的子图
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    # flatten()使axes方便迭代
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            if len(img.shape) == 3:
                img = img.permute(1, 2, 0)
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        # 不显示x轴与y轴
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    if tpye:
        return axes
    else:
        plt.show()


def gray_to_rgb(img):
    return img.convert('RGB')


def load_mnist(batch_size=8, shape=(224, 224)):
    """MNIST"""
    trans = transforms.Compose([gray_to_rgb,
                                transforms.Resize(shape),
                                transforms.ToTensor()])
    train_data = torchvision.datasets.MNIST(root="../dataset", train=True, transform=trans, download=False)
    test_data = torchvision.datasets.MNIST(root="../dataset", train=False, transform=trans, download=False)
    return (data.DataLoader(train_data, batch_size, shuffle=True, num_workers=2),
            data.DataLoader(test_data, batch_size, shuffle=False, num_workers=2))


def load_cifar_10(batch_size=8, shape=(224, 224)):
    """CIFAR-10"""
    trans = transforms.Compose([transforms.Resize(shape),
                                transforms.ToTensor(),
                                transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
                                transforms.RandomVerticalFlip(p=0.5),  # 垂直
                                transforms.RandomRotation(degrees=180),  # 旋转
                                transforms.ColorJitter(brightness=0.5, contrast=0.5),  # 对比度和饱和度
                                transforms.Normalize(mean=[0.229, 0.224, 0.225], std=[0.485, 0.456, 0.406])])
    train_data = torchvision.datasets.CIFAR10(root="../dataset", train=True, transform=trans, download=False)
    test_data = torchvision.datasets.CIFAR10(root="../dataset", train=False, transform=trans, download=False)
    return (data.DataLoader(train_data, batch_size, shuffle=True, num_workers=2),
            data.DataLoader(test_data, batch_size, shuffle=False, num_workers=2))


def load_caltech_256(batch_size=8, shape=(224, 224)):
    """CALTECH-256"""
    trans_test = transforms.Compose([gray_to_rgb,
                                     transforms.Resize(shape),
                                     transforms.ToTensor()])
    trans = transforms.Compose([gray_to_rgb,
                                transforms.Resize(shape),
                                transforms.ToTensor(),
                                transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
                                transforms.RandomVerticalFlip(p=0.5),  # 垂直
                                transforms.RandomRotation(degrees=180),  # 旋转
                                transforms.ColorJitter(brightness=0.5, contrast=0.5),  # 对比度和饱和度
                                transforms.Normalize(mean=[0.229, 0.224, 0.225], std=[0.485, 0.456, 0.406])])

    train_data = torchvision.datasets.Caltech256(root="../dataset", transform=trans, download=False)
    test_data = torchvision.datasets.Caltech256(root="../dataset", transform=trans_test, download=False)
    return (data.DataLoader(train_data, batch_size, shuffle=True, num_workers=2),
            data.DataLoader(test_data, batch_size, shuffle=True, num_workers=2))


def get_caltech_256_label(key):
    if isinstance(key, int):
        return class_name[key]
    return [class_name[k] for k in key]


def get_out_layer(net, in_channels, shape):
    """得到展平"""
    shape = net(torch.randn(size=(1, in_channels, shape[0], shape[1]))).shape[-2:]
    return shape[0] * shape[1]


def test_net(net, shape):
    """测试输出"""
    x = torch.randn(size=(2, 3, shape[0], shape[1]))
    print(net(x).shape)


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def load_net_param(net, path):
    """加载训练的模型"""
    if os.path.exists(path):
        net.load_state_dict(torch.load(path))
        print(f'load model param with {net.name}.pth')
    else:
        net.apply(weight_init)
        print(f'init param')


def train_epoch(net, iter, loss, optimizer, device, mode='train'):
    if mode == 'train':
        net.train()
    else:
        net.eval()
    loss_list = []
    accs_list = []
    for X, y in tqdm(iter):
        X, y = X.to(device), y.to(device)
        y_hat = net(X)
        y_loss = loss(y_hat, y)
        if mode == 'train':
            optimizer.zero_grad()
            y_loss.backward()
            optimizer.step()

        acc = (y_hat.argmax(dim=-1) == y).float().mean()
        # print(f'{y_loss.item()}, {acc.item()}')
        loss_list.append(y_loss.item())
        accs_list.append(acc.item())

    epoch_loss = sum(loss_list) / len(loss_list)
    epoch_acc = sum(accs_list) / len(accs_list)
    return epoch_loss, epoch_acc


def train(net, train_iter, valid_iter, num_epochs, learning_rate, weight_decay, device, path, save=True):
    net.to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    optimizer_adjust = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    # best_acc = 0.0
    # best_acc = 0.88263
    # animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
    #                     legend=['train loss', 'train acc', 'valid loss', 'valid acc'])
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], legend=['train loss', 'train acc'])
    timer = Timer()
    for epoch in range(num_epochs):
        timer.start()
        # train
        train_loss, train_acc = train_epoch(net, train_iter, loss, optimizer, device)
        print(f'[ Train | {epoch + 1:03d}/{num_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f},  '
              f'lr = {optimizer.param_groups[0]["lr"]}')

        # valid
        # valid_loss, valid_acc = train_epoch(net, valid_iter, loss, optimizer, device, mode='eval')
        # print(f'[ Valid | {epoch + 1:03d}/{num_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}')

        timer.stop()

        # learning_rate
        optimizer_adjust.step(train_loss)

        # animator
        # animator.add(epoch + 1, (train_loss, train_acc, valid_loss, valid_acc))
        animator.add(epoch + 1, (train_loss, train_acc))

        # save
        # if valid_acc > best_acc:
        #     best_acc = valid_acc
        #     torch.save(net.state_dict(), path)
        #     print(f'saving model with acc {best_acc:.3f}')
        if save:
            torch.save(net.state_dict(), path)
            print(f'saving model with acc {train_acc:.3f}')

    plt.show()
    print(f'all: {timer.sum():.3f}')


def pred(net, valid_iter, device):
    net.to(device)
    net.eval()

    X, y = next(iter(valid_iter))
    y_pred = net(X).argmax(dim=-1)

    s = []
    for i, j in zip(get_caltech_256_label(y_pred.tolist()), get_caltech_256_label(y.tolist())):
        s.append(f'{i} = {j}')
    show_images(X, 4, 4, s, scale=5)
