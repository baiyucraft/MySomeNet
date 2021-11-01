Config = {
    'num_classes': 20 + 1,
    # 图片大小
    'min_dim': 300,
    'in_height': 300,
    'in_width': 300,

    # 大小
    'sizes': [[0.1, 0.14], [0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79], [0.88, 0.961]],

    # 宽高比
    'ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],

    'variance': [0.1, 0.2],
    'clip': True,
    'Classes': ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
}
