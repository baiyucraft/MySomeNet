import torchvision
from torch import nn
from torch.utils import data

from util import train, weights_init


def get_data(batch_size):
    """获取数据集"""
    trans = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Grayscale(num_output_channels=1),
    ])
    cifar_train = torchvision.datasets.CIFAR10(root="../dataset", train=True, transform=trans, download=False)
    train_iter = data.DataLoader(cifar_train, batch_size=batch_size)
    return train_iter


class SelfAttention(nn.Module):
    def __init__(self, channels, classes, heads=1):
        super(SelfAttention, self).__init__()
        self.mh = nn.MultiheadAttention(channels, heads)
        self.mlp = nn.Sequential(nn.Flatten(), nn.Linear(channels, classes))

    def forward(self, x):
        bs = x.shape[0]
        x = x.reshape(bs, 1, -1)
        x, _ = self.mh(x, x, x)
        x = self.mlp(x)
        return x


class SelfAttention2(nn.Module):
    def __init__(self, channels, classes, heads=1):
        super(SelfAttention2, self).__init__()
        self.mh1 = nn.MultiheadAttention(channels, heads)
        self.mh2 = nn.MultiheadAttention(channels, heads)
        self.mlp = nn.Sequential(nn.Flatten(), nn.Linear(channels, classes))

    def forward(self, x):
        bs = x.shape[0]
        x = x.reshape(bs, 1, -1)
        y, _ = self.mh1(x, x, x)
        x = x + y
        x, _ = self.mh2(x, x, x)
        x = self.mlp(x)
        return x


if __name__ == '__main__':
    # 200
    num_epochs, lr, batch_size = 1000, 0.0008, 1024
    train_iter = get_data(batch_size)

    net, path = SelfAttention(1024, 10, heads=8), 'model_data/att.pt'
    weights_init(net, path)
    train(net, train_iter, num_epochs, lr, path)

    # net, path = SelfAttention2(1024, 10, heads=8), 'model_data/att2.pt'
    # weights_init(net, path)
    # train(net, train_iter, num_epochs, lr, path)

    # net, path = nn.Sequential(nn.Linear(1024, 256), nn.Tanh(),
    #                           nn.Linear(256, 128), nn.Sigmoid(),
    #                           nn.Linear(128, 10)), 'model_data/bp.pt'
    # weights_init(net, path)
    # train(net, train_iter, num_epochs, lr, path)
