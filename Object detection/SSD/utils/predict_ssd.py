import colorsys
import time

import cv2
import torch
from matplotlib import pyplot as plt
from torchvision import transforms
from tqdm import tqdm

from nets.ssd import get_ssd
from utils.config import Config
from utils.utils import show_bboxes


class SSD(object):
    def __init__(self, confidence=0.5, nms_iou=0.45, size=300, device='cpu'):
        self.net, self.colors = None, None
        self.model_path = f'model_data/m_mask_ssd{size}.pth'
        self.confidence = confidence
        self.nms_iou = nms_iou
        self.size = size
        self.input_shape = (size, size)
        self.device = device

        self.class_names = Config['Classes']
        self.num_classes = len(self.class_names) + 1
        self.generate()

    def generate(self):
        """载入模型"""
        net = get_ssd("test", self.num_classes, self.confidence, self.nms_iou)
        net.load_state_dict(torch.load(self.model_path, map_location=self.device))
        net.to(self.device)
        self.net = net.eval()

        print(f'{self.model_path} model, anchors, and classes loaded.')
        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.) for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))

    def get_boxes_(self, image, show=True):
        trans = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize(self.input_shape),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])
        photo = trans(image).unsqueeze(0).to(self.device)

        # 1 * num_classes * top_k * 5
        preds = self.net(photo)

        if show:
            top_conf = []
            top_label = []
            top_bboxes = []

            for cl in range(self.num_classes):
                p = 0
                while preds[0, cl, p, 0] >= self.confidence:
                    score = preds[0, cl, p, 0]
                    label_name = self.class_names[cl - 1]
                    pt = (preds[0, cl, p, 1:]).detach().numpy()
                    coords = [pt[0], pt[1], pt[2], pt[3]]

                    top_conf.append(score)
                    top_label.append(label_name)
                    top_bboxes.append(coords)
                    p += 1

            if len(top_conf) <= 0:
                print('noo')
                return None, None, None

            image_shape = image.shape[:2]
            top_bboxes = torch.Tensor(top_bboxes)
            top_x1, top_x2 = top_bboxes[:, 0] * image_shape[1], top_bboxes[:, 2] * image_shape[1]
            top_y1, top_y2 = top_bboxes[:, 1] * image_shape[0], top_bboxes[:, 3] * image_shape[0]
            boxes = torch.stack((top_x1, top_y1, top_x2, top_y2), dim=1)

            return boxes, top_label, top_conf

    def detect_image(self, image, axes=None):
        """检测图片"""
        boxes, top_label, top_conf = self.get_boxes_(image)
        if boxes is not None:
            show_bboxes(image, boxes, top_label, top_conf, self.colors, axes=axes)
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
            self.get_boxes_(img, show=False)
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


    """
    def get_FPS(self, image, test_interval):
        image_shape = np.array(np.shape(image)[0:2])
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        if self.letterbox_image:
            crop_img = np.array(letterbox_image(image, (self.input_shape[1], self.input_shape[0])))
        else:
            crop_img = image.convert('RGB')
            crop_img = crop_img.resize((self.input_shape[1], self.input_shape[0]), Image.BICUBIC)

        photo = np.array(crop_img, dtype=np.float64)
        with torch.no_grad():
            photo = torch.from_numpy(np.expand_dims(np.transpose(photo - MEANS, (2, 0, 1)), 0)).type(torch.FloatTensor)
            if self.cuda:
                photo = photo.cuda()
            preds = self.net(photo)
            top_conf = []
            top_label = []
            top_bboxes = []
            for i in range(preds.size(1)):
                j = 0
                while preds[0, i, j, 0] >= self.confidence:
                    score = preds[0, i, j, 0]
                    label_name = self.class_names[i - 1]
                    pt = (preds[0, i, j, 1:]).detach().numpy()
                    coords = [pt[0], pt[1], pt[2], pt[3]]
                    top_conf.append(score)
                    top_label.append(label_name)
                    top_bboxes.append(coords)
                    j = j + 1

            if len(top_conf) > 0:
                top_conf = np.array(top_conf)
                top_label = np.array(top_label)
                top_bboxes = np.array(top_bboxes)
                top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:, 0], -1), np.expand_dims(
                    top_bboxes[:, 1], -1), np.expand_dims(top_bboxes[:, 2], -1), np.expand_dims(top_bboxes[:, 3], -1)
                # -----------------------------------------------------------#
                #   去掉灰条部分
                # -----------------------------------------------------------#
                if self.letterbox_image:
                    boxes = ssd_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax,
                                              np.array([self.input_shape[0], self.input_shape[1]]), image_shape)
                else:
                    top_xmin = top_xmin * image_shape[1]
                    top_ymin = top_ymin * image_shape[0]
                    top_xmax = top_xmax * image_shape[1]
                    top_ymax = top_ymax * image_shape[0]
                    boxes = np.concatenate([top_ymin, top_xmin, top_ymax, top_xmax], axis=-1)

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                preds = self.net(photo)
                top_conf = []
                top_label = []
                top_bboxes = []
                for i in range(preds.size(1)):
                    j = 0
                    while preds[0, i, j, 0] >= self.confidence:
                        score = preds[0, i, j, 0]
                        label_name = self.class_names[i - 1]
                        pt = (preds[0, i, j, 1:]).detach().numpy()
                        coords = [pt[0], pt[1], pt[2], pt[3]]
                        top_conf.append(score)
                        top_label.append(label_name)
                        top_bboxes.append(coords)
                        j = j + 1

                if len(top_conf) > 0:
                    top_conf = np.array(top_conf)
                    top_label = np.array(top_label)
                    top_bboxes = np.array(top_bboxes)
                    top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:, 0], -1), np.expand_dims(
                        top_bboxes[:, 1], -1), np.expand_dims(top_bboxes[:, 2], -1), np.expand_dims(top_bboxes[:, 3],
                                                                                                    -1)
                    # -----------------------------------------------------------#
                    #   去掉灰条部分
                    # -----------------------------------------------------------#
                    if self.letterbox_image:
                        boxes = ssd_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax,
                                                  np.array([self.input_shape[0], self.input_shape[1]]), image_shape)
                    else:
                        top_xmin = top_xmin * image_shape[1]
                        top_ymin = top_ymin * image_shape[0]
                        top_xmax = top_xmax * image_shape[1]
                        top_ymax = top_ymax * image_shape[0]
                        boxes = np.concatenate([top_ymin, top_xmin, top_ymax, top_xmax], axis=-1)

        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    """
if __name__ == '__main__':
    ssd = SSD(300)
