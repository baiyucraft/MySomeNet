import os
import torch
from Tools.utils import show_images, train, test_net, try_gpu, load_caltech_256, get_caltech_256_label
from net import AlexNet

# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context
if __name__ == '__main__':
    image_shape = (224, 224)
    num_epochs, lr, weight_decay = 100, 3e-1, 0.9

    # print(os.path.exists(os.path.join('../dataset', "256_ObjectCategories")))
    train_iter, test_iter = load_caltech_256(32)
    # X, y = next(iter(train_iter))
    # show_images(X, 3, 3, get_caltech_256_label(y.tolist()))

    net = AlexNet(257)
    net_path = f'model_data/{net.name}.pth'
    net.load_state_dict(torch.load(net_path))
    # test_net(net, image_shape)
    train(net, train_iter, test_iter, num_epochs, lr, weight_decay, try_gpu(), net_path)
