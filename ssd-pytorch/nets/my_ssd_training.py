import os
import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from utils.box_utils import log_sum_exp, match
from utils.config import Config

MEANS = (104, 117, 123)


class MultiBoxLoss(nn.Module):
    """
    Args:
        num_classes: 种类
        overlap_thresh: iou的阈值
        neg_pos: 负样本与正样本个数的比例
        device: cpu or gpu
    """

    def __init__(self, num_classes, overlap_thresh, neg_pos=3.0, device='cpu'):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.neg_pos_ratio = neg_pos
        self.device = device
        self.variance = Config['variance']

    def forward(self, predictions, targets):
        # 位置置信，类别置信，锚框
        loc_data, conf_data, priors = predictions
        print(loc_data.shape, conf_data.shape, priors.shape)

        batch_size = loc_data.shape[0]
        num_priors = priors.shape[0]

        # 提前创建容器
        loc_t = torch.zeros(batch_size, num_priors, 4)
        conf_t = torch.zeros(batch_size, num_priors).long()

        loc_t = loc_t.to(self.device)
        conf_t = conf_t.to(self.device)
        priors = priors.to(self.device)

        for i in range(batch_size):
            if not len(targets[i]):
                continue
            # 真实框
            truths = targets[i][:, :-1]
            # 标签
            labels = targets[i][:, -1]
            # 默认锚框
            defaults = priors
            # 匹配，得到位置便宜置信和类别置信
            loc_t[i], conf_t[i] = match(self.threshold, truths, defaults, self.variance, labels)

        # 所有conf_t>0的地方，代表内部包含物体，batch_size * num_priors
        pos = conf_t > 0
        # 取出所有的正样本
        loc_p = loc_data[pos]
        loc_t = loc_t[pos]
        # 计算正样本损失，近端二次，远端一次
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # (batch_size * num_priors) * num_classes
        batch_conf = conf_data.reshape(-1, self.num_classes)
        # 难分类(hard Negative Mining)的锚框, -log(softmax)
        conf_log_p = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))
        conf_log_p = conf_log_p.reshape(batch_size, -1)
        # 只考虑负样本
        conf_log_p[pos] = 0
        # loss_c 降序排列 得到loss_idx下标
        _, loss_idx = conf_log_p.sort(1, descending=True)
        # 得到 loss_c 的元素 在降序排列中的下标，batch_size * num_priors
        _, idx_rank = loss_idx.sort(1)

        # 计算正样本数
        num_pos = pos.sum(1, keepdim=True)
        # 限制负样本数量, 值最大为: 锚框数 - 1
        num_neg = (self.neg_pos_ratio * num_pos).clamp(max=num_priors - 1)
        # (batch_size * num_priors) < (batch_size * 1)
        neg = idx_rank < num_neg

        # batch_size * num_priors  =>  batch_size * num_priors * num_classes
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)

        # 选取出用于训练的正样本与负样本，计算loss n * num_classes
        conf_p = conf_data[pos_idx + neg_idx].reshape(-1, self.num_classes)
        # 真实值 [n]
        truth_p = conf_t[pos + neg]
        # 交叉熵
        loss_c = F.cross_entropy(conf_p, truth_p, reduction='sum')

        # 正样本个数
        N = num_pos.sum().float()
        loss_l /= N
        loss_c /= N

        return loss_l, loss_c


class LossHistory:
    def __init__(self, log_dir):
        import datetime
        curr_time = datetime.datetime.now()
        time_str = datetime.datetime.strftime(curr_time, '%Y_%m_%d_%H_%M_%S')
        self.log_dir = log_dir
        self.time_str = time_str
        self.save_path = os.path.join(self.log_dir, "loss_" + str(self.time_str))
        self.losses = []
        self.val_loss = []

        os.makedirs(self.save_path)

    def append_loss(self, loss, val_loss):
        self.losses.append(loss)
        self.val_loss.append(val_loss)
        with open(os.path.join(self.save_path, "epoch_loss_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.save_path, "epoch_val_loss_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth=2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15

            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle='--', linewidth=2,
                     label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle='--', linewidth=2,
                     label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.save_path, "epoch_loss_" + str(self.time_str) + ".png"))


def weights_init(net, init_type='normal', init_gain=0.01):
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

    print(f'initialize network with {init_type} type')
    net.apply(init_func)
