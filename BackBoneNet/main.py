from Tools.utils import *
from net import *
from torchvision import models

# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

if __name__ == '__main__':
    image_shape = (224, 224)
    num_epochs, lr, weight_decay = 50, 1e-3, 5e-4
    # train_iter, test_iter = load_mnist(64)
    # train_iter, test_iter = load_cifar_10(64)
    train_iter, test_iter = load_caltech_256(16)
    # X, y = next(iter(train_iter))
    # show_images(X, 9, 9, get_caltech_256_label(y.tolist()))

    # net = AlexNet(10)
    # net = VGG(10)
    # net = NiN(10)
    # net = GoogLeNet(10)
    # ---
    # net = InceptionV23(257)
    # net = ResNet(257)
    # net = InceptionV4(257)
    # net = InceptionRes(257, mode='V2')
    # net = DenseNet(257)
    # ---
    # net = MobileNet(257)
    # net = MobileNetV2(257)
    # ---
    net = MobileNetV3(257)
    net_path = f'model_data/{net.name}.pth'
    load_net_param(net, net_path)
    train(net, train_iter, test_iter, num_epochs, lr, weight_decay, try_gpu(), net_path, save=True)
    # pred(net, train_iter, 'cpu')
