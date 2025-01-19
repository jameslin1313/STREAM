import cv2 as cv
import os
import json
import numpy as np
import argparse
import shutil


def load_OCR(filename):
    with open(filename, encoding='utf-8') as json_f:
        anno = json.load(json_f)
        
        ocr_results = anno["form"]

        texts = []  # string
        boxes = []  # box of string
        ser_labels = []
        words = []
        ids = []
        links = []

        for i, ocr in enumerate(ocr_results):
            text = ocr['text']
            box = ocr['box']
            ser_label = ocr['label']
            word = ocr['words']
            idx = ocr['id']
            # linking = ocr['linking']

            texts.append(text)
            boxes.append(box)
            ser_labels.append(ser_label)
            words.append(word)
            ids.append(idx)
    
    return texts, boxes, ser_labels, words, ids


def matching(decode_box, anno_boxes, anno_labels, eval_decode=True):
    max_iou, max_overlap = 0, 0
    iou_label, overlap_label = 'Not defined', 'Not defined'

    for i, (box, label) in enumerate(zip(anno_boxes, anno_labels)):
        iou, overlap = IOU(decode_box, box)

        if iou > max_iou:
            max_iou = iou
            iou_label = label
        if overlap > max_overlap:
            max_overlap = overlap
            overlap_label = label
    
    if max_iou > max_overlap:
        criterion = max_iou
        criterion_label = iou_label
    else:
        criterion = max_overlap
        criterion_label = overlap_label
    
    if criterion > 0.6:
        if eval_decode and criterion_label != 'other':  # group with non-other
            return False, 'other'
        elif not eval_decode:
            return True, criterion_label
        else:   # group only other
            return True, 'other'
    else:
        return False, 'error'


def IOU(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    inter_x1 = max(x1, x3)
    inter_y1 = max(y1, y3)
    inter_x2 = min(x2, x4)
    inter_y2 = min(y2, y4)

    # Calculate intersection area
    inter_area = max(0, inter_x2 - inter_x1 + 1) * max(0, inter_y2 - inter_y1 + 1)

    # Calculate area of each bounding box
    area_box1 = (x2 - x1 + 1) * (y2 - y1 + 1)
    area_box2 = (x4 - x3 + 1) * (y4 - y3 + 1)
    
    # IOU
    union_area = area_box1 + area_box2 - inter_area

    iou = inter_area / union_area if union_area != 0 else 0

    # check overlap ratio
    overlap1 = inter_area / area_box1
    overlap2 = inter_area / area_box2

    if overlap1 > overlap2:
        return iou, overlap1
    
    return iou, overlap2


def main(args):
    base_data_path = '/media/ai2lab/4TB SSD/Datasets/'
    if args.dataset == 'FUNSD':
        img_dir = os.path.join(base_data_path + 'FUNSD/my-FUNSD/grouping/' + args.data_type + 'ing_data/images')
        anno_dir = os.path.join(base_data_path + 'FUNSD/my-FUNSD/classification/' + args.data_type + 'ing_data/annotations')
    elif args.dataset == 'XFUND-r':
        img_dir = os.path.join(base_data_path + args.dataset + '/classification/' + args.data_type + 'ing_data-v2/images')
        anno_dir = os.path.join(base_data_path + args.dataset + '/classification/' + args.data_type + 'ing_data-v2/annotations')
    elif args.dataset == 'TransGlobe':
        # data_type: Medical, Medical-Receipt
        if args.data_type == 'Medical' or args.data_type == 'Medical-Receipt':
            img_dir = '/media/ai2lab/4TB SSD/Datasets/TransGlobe/' + args.data_type + '/images'
            anno_dir = '/media/ai2lab/4TB SSD/Datasets/TransGlobe/' + args.data_type + '/OCRs'
        else:
            print('Data Type of TransGlobe Should Be "Medical" or "Medical-Receipt" !')
            exit(-1)
    else:
        img_dir = os.path.join(base_data_path + args.dataset + '/classification/' + args.data_type + 'ing_data/images')
        anno_dir = os.path.join(base_data_path + args.dataset + '/classification/' + args.data_type + 'ing_data/annotations')
    

    decode_dir = '/home/ai2lab/Desktop/VDU/LayoutLM_GP/outputs/results/' + args.dataset + '-decode/' + args.data_type + 'ing_data'
    if args.mode == 'OCR':
        decode_dir = '/home/ai2lab/Desktop/VDU/Azure/OCR-results/' + args.dataset + '/' + args.data_type

    img_name = sorted(os.listdir(img_dir))
    anno_name = sorted(os.listdir(anno_dir))
    decode_name = sorted(os.listdir(decode_dir))
    
    draw_dir = 'visualize/eval-grouping/' + args.dataset + '/' + args.data_type
    if os.path.exists(draw_dir):
        shutil.rmtree(draw_dir)  # If it exists, remove it along with its contents
    os.makedirs(draw_dir)

    if args.save_json:
        if args.mode == 'OCR':
            save_dir = 'outputs/results/' + args.dataset + '-decode/OCR/' + args.data_type + 'ing_data'
        else:
            save_dir = f"outputs/results/{args.dataset}-decode/effective/{args.data_type}ing_data/annotations"
        
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)  # If it exists, remove it along with its contents
        os.makedirs(save_dir)

    # color = (0, 0, 255)  # Red color in BGR
    color = {'header': (255,0,255), 'question': (255,0,0), 'answer': (0,255,0), 'other': (0,255,255), 'error': (0,0,255), '<BLANK>': (0,165,255), 'Not defined': (0, 165, 255)}
    thickness = 2


    if args.data_range == -1:
        data_range = range(len(anno_name))
    else:
        data_range = range(args.data_range)

    full_correct, full_anno_len, full_decode_len = 0, 0, 0
    empty_type = '<BLANK> '
    
    for i in data_range:
        print(i, img_name[i], anno_name[i], decode_name[i])

        img_file = os.path.join(img_dir, img_name[i])
        anno_file = os.path.join(anno_dir, anno_name[i])
        decode_file = os.path.join(decode_dir, decode_name[i])

        draw_file = os.path.join(draw_dir, img_name[i])
        if args.save_json:
            save_file = os.path.join(save_dir, anno_name[i])

        # load annotations
        anno_texts, anno_boxes, anno_labels, anno_words, anno_ids = load_OCR(anno_file)
        # load prediction
        decode_texts, decode_boxes, decode_labels, decode_words, decode_ids = load_OCR(decode_file)

        correct, blank = 0, 0
        form = []
        for j, (text, box, label, words, idx) in enumerate(zip(decode_texts, decode_boxes, decode_labels, decode_words, decode_ids)):
            if text == empty_type:
                blank += 1
                continue
            if label == 'error':
                # print(text, box, label)
                other_type, _ = matching(box, anno_boxes, anno_labels)
                if other_type:
                    decode_labels[j] = 'other'
                    label = 'other'
                    correct += 1
                else:
                    replaced_type = 'other'
                    # replaced_type = 'header'
                    decode_labels[j] = replaced_type  # use 'header' type to let this entity wrong when doing SER
                    label = replaced_type
                # print('-----------------------------------')
            elif label == 'Not defined':
                matched, matched_label = matching(box, anno_boxes, anno_labels, eval_decode=False)
                
                # decode_labels[j] = matched_label
                # label = matched_label

                decode_labels[j] = 'question'  # can not use 'other' type !
                label = 'question'

                if matched:
                    correct += 1
            else:
                correct += 1
            
            ocr = {"text": text, "box": box, "label": label, "words": words, "id": idx, "linking": []}
            form.append(ocr)
            # print(text, box, decode_labels[i])
        
        full_correct += correct
        full_anno_len += len(anno_labels) - blank
        full_decode_len += len(decode_labels) - blank
        
        # draw image
        image = cv.imread(img_file)
        for (t, b, label) in zip(decode_texts, decode_boxes, decode_labels):
            if t == empty_type:
                continue
            cv.rectangle(image, (b[0], b[1]), (b[2], b[3]), color[label], thickness)
        
        cv.imwrite(draw_file, image)
        
        # write new annotation
        if args.save_json:
            funsd = {'form': form}
            with open(save_file, 'w', encoding='utf-8') as f:
                json.dump(funsd, f, ensure_ascii=False, indent=4)

        
    avg_ent_precision = full_correct / full_decode_len
    avg_ent_recall = full_correct / full_anno_len
    avg_ent_f1 = 2 * avg_ent_precision * avg_ent_recall / (avg_ent_precision + avg_ent_recall)
    print("==========================================")
    print(f'avg_precision: {avg_ent_precision}, avg_recall: {avg_ent_recall}, avg_f1: {avg_ent_f1}')
    print("==========================================")
        
                



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='FUNSD')
    parser.add_argument('--data_type', type=str, default='test')
    parser.add_argument('--data_range', type=int, default=-1)  # -1 indicate all examples
    parser.add_argument('--save_json', action='store_false')   # default is True
    parser.add_argument('--mode', type=str, default='decode')  # decode: LayoutLM_GP output / OCR: OCR performance eval
    args = parser.parse_args()

    main(args)