import cv2 as cv
import os
import json
import numpy as np
import argparse

color = {
    'menu.nm': (255, 255, 255),                  # Red
    'menu.cnt': (60, 180, 75),                   # Green
    'menu.price': (255, 225, 25),                # Yellow
    'other': (0, 130, 200),                      # Blue
    'sub_total.subtotal_price': (245, 130, 48),  # Orange
    'sub_total.service_price': (145, 30, 180),   # Purple
    'sub_total.tax_price': (70, 240, 240),       # Cyan
    'sub_total.etc': (240, 50, 230),             # Magenta
    'total.total_price': (210, 245, 60),         # Lime
    'menu.sub_nm': (250, 190, 190),              # Pink
    'total.cashprice': (0, 128, 128),            # Teal
    'total.changeprice': (230, 190, 255),        # Lavender
    'total.menutype_cnt': (170, 110, 40),        # Brown
    'total.menuqty_cnt': (255, 250, 200),        # Beige
    'sub_total.discount_price': (128, 0, 0),     # Maroon
    'menu.unitprice': (170, 255, 195),           # Mint
    'total.total_etc': (128, 128, 0),            # Olive
    'total.creditcardprice': (255, 215, 180),    # Coral
    'menu.num': (0, 0, 128),                     # Navy
    'menu.sub_cnt': (128, 128, 128),             # Grey
    'menu.discountprice': (255, 165, 0),         # Gold
    'menu.sub_price': (173, 216, 230),           # Light Blue
    'total.emoneyprice': (123, 104, 238),        # Medium Purple
    'menu.sub_unitprice': (34, 139, 34),         # Forest Green
    'void_menu.nm': (255, 69, 0),                # Red-Orange
    'void_menu.price': (199, 21, 133),           # Deep Pink
    'sub_total.othersvc_price': (255, 20, 147),  # Hot Pink
    'menu.vatyn': (32, 178, 170),                # Light Sea Green
    'menu.itemsubtotal': (84, 86, 255),          # Indigo
    'menu.etc': (85, 87, 128),                   # Steel Grey
    'error': (0,0,255)                           # Red
}

thickness = 2


def load_OCR(path, name=None, chatgpt=False):
    json_file = os.path.join(path, name)

    with open(json_file, encoding='utf-8') as json_f:
        anno = json.load(json_f)
        
        ocr_results = anno["valid_line"]

        full_words = []  # string
        full_boxes = []  # box
        categories = []  # ser label

        for i, ocr in enumerate(ocr_results):
            words = ocr['words']
            texts = ''
            boxes = []
            for word in words:
                box = word['quad']
                text = word['text']

                if len(texts) == 0:
                    texts = text
                else:
                    texts += text + ' '
                boxes.append([box['x1'], box['y1'], box['x3'], box['y3']])
            
            category = ocr['category']

            if len(boxes) <= 0:
                continue
            
            full_words.append(texts)
            x1, x2 = min(row[0] for row in boxes), max(row[2] for row in boxes)
            y1, y2 = min(row[1] for row in boxes), max(row[3] for row in boxes)
            box = [x1, y1, x2, y2]
            full_boxes.append(box)
            categories.append(category)
    
    return full_boxes, categories


def main(args):
    base_data_path = '/media/ai2lab/4TB SSD/Datasets/'
    data_path = os.path.join(base_data_path, 'CORD-r', 'classification', args.data_type)
    img_dir = os.path.join(data_path, 'image')

    if args.anno_mode == 'decode':
        anno_dir = '/home/ai2lab/Desktop/VDU/LayoutLM_GP/outputs/results/CORD-r-decode/' + args.data_type + 'ing_data'
    else:
        anno_dir = os.path.join(data_path, 'json')

    img_name = sorted(os.listdir(img_dir))
    anno_name = sorted(os.listdir(anno_dir))

    draw_dir = '/home/ai2lab/Desktop/VDU/LayoutLM_GP/visualize/CORD-r-visualize/draw-' + args.data_type
    if not os.path.exists(draw_dir):
            os.makedirs(draw_dir)
    thickness = 2
    
    if args.data_range == -1:
        data_range = range(len(anno_name))
    else:
        data_range = range(args.data_range)

    count = 0
    categories = []

    for i in data_range:
        print(i, img_name[i], anno_name[i])
        boxes, ser_labels = load_OCR(anno_dir, anno_name[i])
        count += len(boxes)
        
        for label in ser_labels:
            if label not in categories:
                categories.append(label)
        
        image = cv.imread(os.path.join(img_dir, img_name[i]))
        # print(image.shape)
        # # image = cv.resize(image, (1000,1000))
        for (b, label) in zip(boxes, ser_labels):
            if label == 'error':
                print('error', anno_name[i])
            cv.rectangle(image, (b[0], b[1]), (b[2], b[3]), color[label], thickness)

        cv.imwrite(os.path.join(draw_dir, img_name[i]), image)        
    print(count)
    # print(categories)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, default='test')
    parser.add_argument('--data_range', type=int, default=-1)  # -1 indicate all examples
    parser.add_argument('--anno_mode', type=str, default='annotation')  # visualize / decode / effective-decode / layoutlmv3 / 

    args = parser.parse_args()

    main(args)