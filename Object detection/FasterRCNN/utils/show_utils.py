import torch
from matplotlib import pyplot as plt

Classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
           'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


def bbox_to_rect(bbox, color):
    # 将边界框 (左上x, 左上y, 右下x, 右下y) 格式转换成 matplotlib 格式：
    # ((左上x, 左上y), 宽, 高)
    return plt.Rectangle(
        xy=(bbox[0], bbox[1]),
        width=bbox[2] - bbox[0],
        height=bbox[3] - bbox[1],
        fill=False, edgecolor=color, linewidth=2)


def show_bboxes(image, bboxes, labels=None, confs=None, colors=None):
    """显示所有边界框。"""
    # h * w * 3
    axes = plt.imshow(image).axes
    # 不显示x轴与y轴
    axes.get_xaxis().set_visible(False)
    axes.get_yaxis().set_visible(False)

    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    box_labels = _make_list([f'{label}={conf:.2f}' for label, conf in zip(labels, confs)])
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        bbox[:2] = bbox[:2] - 5
        bbox[2:] = bbox[2:] + 5
        bbox[:2].clamp_(min=0)
        bbox[2].clamp_(max=image.shape[1])
        bbox[3].clamp_(max=image.shape[0])

        # 单个
        # color = colors[i % len(colors)]
        color = colors[Classes.index(labels[i])]
        rect = bbox_to_rect(bbox.detach().numpy(), color)
        axes.add_patch(rect)
        if box_labels and len(box_labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], box_labels[i], va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))


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
