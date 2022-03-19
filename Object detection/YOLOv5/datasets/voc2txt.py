import xml.etree.ElementTree as ET
import glob


def convert(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def convert_annotation(image_name):
    with open('./pest_test/Annotations/' + image_name[:-3] + 'xml') as f:
        xml_text = f.read()
        root = ET.fromstring(xml_text)

    with open('./pest_test/labels/' + image_name[:-3] + 'txt', 'w') as f:
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls not in classes:
                print(cls)
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            bb = convert((w, h), b)
            f.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


if __name__ == '__main__':
    classes = ["bug", "fruit bug"]
    for image_path in glob.glob("./pest_test/JPEGImages/*.jpg"):  # 每一张图片都对应一个xml文件这里写xml对应的图片的路径
        image_name = image_path.split('\\')[-1]
        convert_annotation(image_name)
