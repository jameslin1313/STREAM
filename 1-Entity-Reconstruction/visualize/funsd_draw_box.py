import cv2
import os
import json
import numpy as np
import argparse
import shutil
from PIL import Image, ImageDraw, ImageFont
import re



def load_OCR(anno_file, blank=False):
    with open(anno_file, encoding='utf-8') as json_f:
        anno = json.load(json_f)
        
        ocr_results = anno["form"]

        texts = []  # string
        boxes = []  # box of string
        centers = []
        ser_labels = []
        links = []

        prompt_ocr = ''

        for i, ocr in enumerate(ocr_results):
            text = ocr['text']
            box = ocr['box']
            ser_label = ocr['label']
            # linking = ocr['linking']

            # if name == '87086073.json':
            #     print(text, box, ser_label)
            texts.append(text)
            boxes.append(box)
            ser_labels.append(ser_label)
    
    return texts, boxes, ser_labels
    


def draw_chinese_text(image, text, position, box_size, font_path, color):
    # Convert the OpenCV image to a PIL image
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    
    # Calculate the maximum font size that fits within the bounding box
    box_width, box_height = box_size
    font_size = 1
    font = ImageFont.truetype(font_path, font_size)
    text_width, text_height = draw.textsize(text, font=font)
    
    while text_width < box_width*0.9 and text_height < box_height*0.9:
        font_size += 1
        font = ImageFont.truetype(font_path, font_size)
        text_width, text_height = draw.textsize(text, font=font)
    
    # Use the largest font size that still fits within the bounding box
    font_size -= 1
    font = ImageFont.truetype(font_path, font_size)
    text_width, text_height = draw.textsize(text, font=font)
    
    # Calculate the position to center the text
    text_x = position[0] + (box_width - text_width) // 2
    text_y = position[1] + (box_height - text_height) // 2
    
    # Draw the Chinese text
    draw.text((text_x, text_y), text, font=font, fill=color)
    
    # Convert the PIL image back to an OpenCV image
    image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    return image


def masking(image, box):
    pass


def inpainting(image, box):
    mask = np.zeros(image.shape[:2], np.uint8)
    # 在掩码上绘制白色矩形
    mask[box[1]:box[3], box[0]:box[2]] = 255
    # 使用 inpaint 方法修复图像
    inpainted_image = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

    return inpainted_image


def clone(image, box):
    x, y, w, h = box[0], box[1], box[2] - box[0], box[3] - box[1]

    src_x, src_y, src_w, src_h = 0, 0 if y - h > 0 else 0, w, h
    # 从源区域复制块
    src_region = image[src_y:src_y+src_h, src_x:src_x+src_w]
    # 创建掩码
    mask = 255 * np.ones(src_region.shape, src_region.dtype)
    # 目标区域中心
    center = (x + w // 2, y + h // 2)
    # 无缝克隆
    cloned_image = cv2.seamlessClone(src_region, image, mask, center, cv2.NORMAL_CLONE)

    return cloned_image



def main(args):
    base_data_path = '/media/ai2lab/4TB SSD/Datasets/'
    if args.dataset == 'FUNSD':
        img_dir = os.path.join(base_data_path + 'FUNSD/my-FUNSD/grouping/' + args.data_type + 'ing_data/images')
    elif args.dataset == 'synthetic':
        img_dir = '/media/ai2lab/4TB SSD/Datasets/Synthetic-form/images'
    elif args.dataset == 'XFUND-r':
        img_dir = os.path.join(base_data_path + args.dataset + '/grouping/' + args.data_type + 'ing_data-v2/images')
    elif args.dataset == 'TransGlobe':
        img_dir = os.path.join(base_data_path + args.dataset + '/Dataset-filled/' + args.data_type + 'ing_data/images')
    elif args.dataset == 'TransGlobe-r':
        img_dir = '/media/ai2lab/4TB SSD/Datasets/TransGlobe/Dataset-filled/testing_data/images'
    elif args.dataset == 'CORD-r':
        img_dir = os.path.join(base_data_path + args.dataset + '/classification/' + args.data_type + '/image')
    else:
        img_dir = os.path.join(base_data_path + args.dataset + '/grouping/' + args.data_type + 'ing_data/images')
        

    # data_path = os.path.join(base_data_path + 'LayoutLM_GP/FUNSD-r/' + args.data_type + "ing_data")
    # anno_dir = 'annotations'

    # img_dir = '/home/ai2lab/Desktop/VDU/LayoutLM_GP/visualize/' + args.dataset + '-visualize/true-images/draw-' + args.data_type

    if args.anno_mode == 'visualize':
        anno_dir = '/home/ai2lab/Desktop/VDU/LayoutLM_GP/visualize/' + args.dataset + '-visualize/' + args.data_type
    elif args.anno_mode == 'effective-decode':
        anno_dir = '/home/ai2lab/Desktop/VDU/LayoutLM_GP/outputs/results/' + args.dataset + '-decode/effective/' + args.data_type + 'ing_data/annotations'
        # anno_dir = '/home/ai2lab/Desktop/VDU/LayoutLM_GP/outputs/results/' + args.dataset + '-decode/OCR/' + args.data_type + 'ing_data'
    elif args.anno_mode == 'decode':
        anno_dir = '/home/ai2lab/Desktop/VDU/LayoutLM_GP/outputs/results/' + args.dataset + '-decode/' + args.data_type + 'ing_data'
    elif args.anno_mode == 'layoutlmv3':
        anno_dir = '/home/ai2lab/Desktop/VDU/unilm/layoutlmv3/eval/' + args.dataset + '/XFUND-format/visualize-anno'
    elif args.anno_mode == 'synthetic':
        anno_dir = '/media/ai2lab/4TB SSD/Datasets/Synthetic-form/annotations'
    elif args.anno_mode == 'annotation':
        anno_dir = '/media/ai2lab/4TB SSD/Datasets/' + args.dataset + '/classification/' + args.data_type + 'ing_data/annotations'
    elif args.anno_mode == 'dataset2r_version':
        anno_dir = '/home/ai2lab/Desktop/VDU/E.Sun dataset/datasets/' + args.dataset + '-r/FUNSD-format/' + args.data_type + 'ing_data/annotations'
    else:
        print('Error anno_mode !!')
        exit(0)

    if args.draw_text:
        anno_dir = base_data_path + 'ESun/classification/' + args.data_type + 'ing_data/annotations'
        blank_dir = base_data_path + 'ESun-blank/classification/' + args.data_type + 'ing_data/annotations'
        font_path = '/home/ai2lab/Documents/Noto_Sans_TC/NotoSansTC-VariableFont_wght.ttf'

        blank_name = sorted(os.listdir(blank_dir))
    
    img_name = sorted(os.listdir(img_dir))
    anno_name = sorted(os.listdir(anno_dir))
    print(anno_dir)
    print(len(img_name), len(anno_name))

    # draw_dir = os.path.join(base_data_path, 'LayoutLM_GP/FUNSD-r/' + args.data_type + 'ing_data/images')
    draw_dir = '/home/ai2lab/Desktop/VDU/LayoutLM_GP/visualize/' + args.dataset + '-visualize/draw-' + args.data_type
    # draw_dir = '/home/ai2lab/Desktop/VDU/LayoutLM_GP/outputs/results/' + args.dataset + '-decode/draw-' + args.data_type
    # draw_dir = '/home/ai2lab/Desktop/VDU/unilm/layoutlmv3/eval/' + args.dataset + '/XFUND-format/visualize'
    # draw_dir = '/home/ai2lab/Desktop/VDU/LayoutLM_GP/visualize/' + args.dataset + '-visualize/'

    if os.path.exists(draw_dir):
        shutil.rmtree(draw_dir)  # If it exists, remove it along with its contents
    os.makedirs(draw_dir)
    
    # color = (0, 0, 255)  # Red color in BGR
    color = {'header': (255,0,255), 'question': (255,0,0), 'answer': (0,255,0), 'other': (0,255,255), 'error': (0,0,255), 'ser-error': (0,165,255), '<BLANK>': (0,0,0)}
    thickness = 2
    
    if args.data_range == -1:
        data_range = range(len(anno_name))
    else:
        data_range = range(args.data_range)

    # img_name = ['image_028.png']
    # anno_name = ['image_028.json']
    # blank_name = ['image_028.json']
    count = 0
    for i in data_range:
        # if args.data_range == 1:
            # img_name[i] = '87594142_87594144.png'
        print(i, img_name[i], anno_name[i])

        img_file = os.path.join(img_dir, img_name[i])
        anno_file = os.path.join(anno_dir, anno_name[i])
        
        texts, boxes, ser_labels = load_OCR(anno_file)
        
        count += len(texts)
        continue

        if args.draw_text:
            blank_file = os.path.join(blank_dir, blank_name[i])
            blank_texts, _, _ = load_OCR(blank_file)

        # print(len(blank_texts), len(texts))

        # for t in range(len(blank_texts)):
        #     print(blank_texts[t])
        # if len(blank_texts) != len(texts):
        #     print('error : ', anno_name[i])
        
        
        image = cv2.imread(img_file)
        # print(image.shape)
        # image = cv2.resize(image, (1000,1000))
        for idx, (box, label) in enumerate(zip(boxes, ser_labels)):
            if label != 'other':
                cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color[label], thickness)

            if args.draw_text:
                if blank_texts[idx] == '<BLANK>':
                    print(texts[idx])
                    center_x = (box[0] + box[2]) // 2
                    center_y = (box[1] + box[3]) // 2

                    box_width = box[2] - box[0]
                    box_height = box[3] - box[1]

                    # image = inpainting(image, box)
                    # image = clone(image, box)
                    image[box[1]:box[3], box[0]:box[2]] = image[0,0]


                    # Put the text on the image
                    # cv2.putText(image, texts[idx], (text_x, text_y), font, font_scale, color=(0, 0, 0), thickness=2)
                    # image = draw_chinese_text(image, texts[idx], (box[0], box[1]), (box_width, box_height), font_path, color=(0, 0, 0))

        draw_file = os.path.join(draw_dir, img_name[i])
        print(draw_file)
        cv2.imwrite(draw_file, image)

    print(count)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ESun')
    parser.add_argument('--data_type', type=str, default='test')
    parser.add_argument('--data_range', type=int, default=-1)  # -1 indicate all examples
    parser.add_argument('--draw_text', action='store_true')
    parser.add_argument('--anno_mode', type=str, default='visualize')  # visualize / decode / effective-decode / layoutlmv3 / 

    parser.add_argument('--data', type=str, default='annotations')
    args = parser.parse_args()

    main(args)

    # directory = os.path.join("/media/ai2lab/4TB SSD/Datasets/XFUND-r", args.anno_mode, f"{args.data_type}ing_data", args.data)

    # if args.data == 'annotations':
    #     filetype = 'json'
    # else:
    #     filetype = 'png'

    # for filename in sorted(os.listdir(directory)):
    #     print(filename)

    #     if args.data_type == 'test':
    #         match = re.match(r"zh_val_(\d+)\."+filetype, filename)
    #     else:
    #         match = re.match(r"zh_train_(\d+)\."+filetype, filename)
    #     if match:
    #         idx = int(match.group(1))
    #         print(idx)

    #         # 生成新的檔名
    #         if args.data_type == 'test':
    #             new_filename = f"zh_val_{idx:02d}."+filetype
    #         else:
    #             new_filename = f"zh_train_{idx:03d}."+filetype
            
    #         # 獲取完整路徑
    #         old_path = os.path.join(directory, filename)
    #         new_path = os.path.join(directory, new_filename)
    #         # 重新命名檔案
    #         os.rename(old_path, new_path)
    #         print(f"Renamed: {filename} -> {new_filename}")
    #         # input()