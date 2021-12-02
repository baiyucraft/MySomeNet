import colorsys
import time

import cv2
import torch
from matplotlib import pyplot as plt
from torchvision import transforms
from tqdm import tqdm

from nets.YOLOv3 import YOLOv3
from utils.config import Config
from utils.show_utils import show_bboxes


class Yolo:
    def __init__(self, device='cpu'):
        self.net, self.colors = None, None
        self.model_path = 'model_data/net.params'
        self.device = device

        self.class_names = Config['Classes']
        self.num_classes = len(self.class_names) + 1
        self.generate()

    def generate(self):
        """载入模型"""
        net = YOLOv3()
        net.load_state_dict(torch.load(self.model_path, map_location=self.device))
        net.to(self.device)
        self.net = net.eval()

        print(f'{self.model_path} model, anchors, and classes loaded.')
        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.) for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))

    def get_boxes_(self, image):
        trans = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize(Config['Size']),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        photo = trans(image).unsqueeze(0)

        preds = self.net(photo, device=self.device)
        if not preds:
            print('no no no!')
            return None, None, None

        boxes = preds[0]['boxes']
        labels = preds[0]['labels']
        scores = preds[0]['scores']

        image_shape = image.shape[:2]
        x1, x2 = boxes[:, 0] * image_shape[1], boxes[:, 2] * image_shape[1]
        y1, y2 = boxes[:, 1] * image_shape[0], boxes[:, 3] * image_shape[0]
        boxes = torch.stack((x1, y1, x2, y2), dim=1)
        return boxes.cpu().detach(), labels.cpu().detach(), scores.cpu().detach()

    def detect_image(self, image, axes=None):
        """检测图片"""
        boxes, labels, scores = self.get_boxes_(image)
        if boxes is not None:
            show_bboxes(image, boxes, labels.tolist(), scores.tolist(), self.colors, axes=axes)
        if axes is None:
            plt.show()

    def show_batch_ssd(self, image_list):
        """批量图片展示"""
        num_rows, num_cols, scale = 1, 5, 10
        figsize = (num_rows * scale, num_rows * scale)
        _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
        axes = axes.flatten()

        for ax, img in zip(axes, image_list):
            self.detect_image(img, axes=ax)
        plt.show()

    def get_fps(self, image_list):
        """计算fps"""
        num = len(image_list)
        start_time = time.time()
        for img in image_list:
            self.get_boxes_(img)
        use_time = time.time() - start_time

        print(f'{num} img use: {use_time:.3f}s, fps: {num / use_time:.3f}')

    def cal_detection_results(self, img_path_list):
        """计算预测结果"""
        for img_name, img_path in tqdm(img_path_list):
            with open(f'mAP/detection_result{self.size}/{img_name}.txt', 'w') as f:
                img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                boxes, top_label, top_conf = self.get_boxes_(img)
                for i, c in enumerate(top_label):
                    predicted_class = c
                    score = top_conf[i]
                    x1, y1, x2, y2 = boxes[i]

                    f.write(f'{predicted_class} {score:.5f} {x1:.0f} {y1:.0f} {x2:.0f} {y2:.0f}\n')
