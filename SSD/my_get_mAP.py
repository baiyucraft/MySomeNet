import os

from bs4 import BeautifulSoup

from utils.predict_ssd import SSD
from utils.utils import try_gpu


def get_dr(size):
    """detection result"""
    ssd = SSD(confidence=0.01, size=size, device=try_gpu())
    img_path = 'dataset/FaceMask/images'

    img_path_list = []
    for img in os.listdir(img_path):
        img_path_list.append((img.split('.')[-2], os.path.join(img_path, img)))
    ssd.cal_detection_results(img_path_list)

    print("Conversion completed!")


def get_gt():
    """ground truth"""
    ann_path = 'dataset/FaceMask/annotations'
    ann_path_list = []
    for ann in os.listdir(ann_path):
        ann_path_list.append((ann.split('.')[-2], os.path.join(ann_path, ann)))

    for ann_name, ann_path in ann_path_list:
        with open(f'mAP/ground_truth/{ann_name}.txt', 'w') as f_txt:
            with open(ann_path) as f_ann:
                soup = BeautifulSoup(f_ann.read(), 'xml')
                objects = soup.find_all('object')
                for ob in objects:
                    x1 = int(ob.find('xmin').text)
                    y1 = int(ob.find('ymin').text)
                    x2 = int(ob.find('xmax').text)
                    y2 = int(ob.find('ymax').text)
                    cls = ob.find('name').text
                    f_txt.write(f'{cls} {x1:.0f} {y1:.0f} {x2:.0f} {y2:.0f}\n')

    print("Conversion completed!")


if __name__ == '__main__':
    get_dr(512)
    # get_gt()
