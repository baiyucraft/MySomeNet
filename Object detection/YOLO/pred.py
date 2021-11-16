import os

import cv2

from utils.pred_yolo import Yolo
from utils.show_utils import try_gpu

if __name__ == '__main__':
    img_path = 'img'
    yolo = Yolo(try_gpu())

    img_path_list = []
    for img in os.listdir(img_path):
        img_path_list.append(os.path.join(img_path, img))

    image_list = [cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB) for img in img_path_list]

    for image in image_list:
        r_image = yolo.detect_image(image)
    # ssd.show_batch_ssd(image_list[:5])
    # ssd.get_fps(image_list*2)
