import json
import cv2
import sys
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import os
import numpy as np



def write_xfund_classificatino(documents, img_dir, save_dir=None, error=False):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    xfund = {'lang': 'zh', 'version': '0.1', 'split': 'val'}
    docs = []
    img_names = sorted(os.listdir(img_dir))
    img_type = img_names[0][-3:]
    print('image data type:', img_type)

    for i in range(len(documents)):
        # get image size
        idx = i
        
        img_file = os.path.join(img_dir, img_names[idx])
        # img_file = os.path.join(img_dir, img_names[i])
        image = cv2.imread(img_file)
        h, w = image.shape[:2]

        img = {'fname': img_names[idx][:-3] + img_type, 'width': w, 'height': h}
        doc = {'id': img_names[idx][:-4], 'uid': img_names[idx][:-3] + img_type, 'document': documents[i], 'img': img}

        docs.append(doc)
    
    # write train set    
    xfund['documents'] = docs
    
    if error:
        # save_path = os.path.join(save_dir, 'zh.val_error.json')
        save_path = os.path.join(save_dir, 'zh.train.json')
    else:
        save_path = os.path.join(save_dir, 'zh.val.json')
    
    print(save_path)
    with open(save_path, 'w', encoding='utf8') as f:
        json.dump(xfund, f, ensure_ascii=False, indent=4)
    
    # save_path = os.path.join(save_dir, 'zh.val.json')
    # print(save_path)
    # with open(save_path, 'w', encoding='utf8') as f:
    #     json.dump(xfund, f, ensure_ascii=False, indent=4)



def main(args):
    base_data_dir = '/media/ai2lab/4TB SSD/Datasets/' + args.dataset

    if args.dataset == 'XFUND-crop':
        img_dir = os.path.join(base_data_dir, 'dataset/' + args.data_type + 'ing_data/images')
    elif args.dataset == 'FUNSD':
        img_dir = os.path.join(base_data_dir, 'my-FUNSD/classification/' + args.data_type + 'ing_data/images')
    elif args.dataset == 'syn_data_fin':
        version_name = sorted(os.listdir(base_data_dir))[args.version_id]
        img_dir = os.path.join(base_data_dir, version_name, 'images')
    elif args.dataset == 'XFUND-r':
        img_dir = os.path.join(base_data_dir, 'grouping/' + args.data_type + 'ing_data-v2/images')
    elif args.dataset == 'TransGlobe':
        # OCR info
        if args.data_type == 'Medical' or args.data_type == 'Medical-Receipt':   # data_type: Medical, Medical-Receipt
            img_dir = '/media/ai2lab/4TB SSD/Datasets/TransGlobe/' + args.data_type + '/images'
        
        # annotated (and filled) info
        elif args.data_type == 'test' and args.version_id == -1:
            base_data_dir = '/media/ai2lab/4TB SSD/Datasets/TransGlobe/Dataset-filled/testing_data'
            img_dir = os.path.join(base_data_dir, 'images')
        
        # leave-one-out XFUND format construction
        elif args.data_type == 'test' or args.data_type == 'train':
            base_data_dir = os.path.join('/media/ai2lab/4TB SSD/Datasets/TransGlobe-r/Azure-OCR/leave-one-out', f"setting_{args.version_id:02}", args.data_type + 'ing_data')
            img_dir = os.path.join(base_data_dir, 'images')
        else:
            print('Data Type of TransGlobe Should Be "Medical" or "Medical-Receipt" !')
            exit(-1)
    else:
        img_dir = os.path.join(base_data_dir, 'classification/' + args.data_type + 'ing_data/images')
    
    anno_dir = 'outputs/results/' + args.dataset + '-decode/effective/' + args.data_type + 'ing_data/annotations'  # no error
    # anno_dir = 'outputs/results/' + args.dataset + '-decode/' + args.data_type + 'ing_data'              # have error
    # anno_dir = 'outputs/results/' + args.dataset + '-decode/OCR/' + args.data_type + 'ing_data'  # OCR only
    # anno_dir = os.path.join(base_data_dir, 'annotations')
    # anno_dir = os.path.join(base_data_dir, version_name, 'annotations')
    anno_dir = '/media/ai2lab/4TB SSD/Datasets/FUNSD/dataset/' + args.data_type + 'ing_data/annotations'
    # anno_dir = '/media/ai2lab/4TB SSD/Datasets/TransGlobe/' + args.data_type + '/OCRs'
    if args.dataset == 'TransGlobe':
        anno_dir = os.path.join(base_data_dir, 'annotations')

    img_names = sorted(os.listdir(img_dir))
    anno_names = sorted(os.listdir(anno_dir))

    xfund_dir = os.path.join(base_data_dir, 'Eval/XFUND-format')
    if args.dataset == 'TransGlobe':
        xfund_dir = "/media/ai2lab/4TB SSD/Datasets/TransGlobe/Eval/XFUND-format"
        xfund_dir = os.path.join(xfund_dir, args.data_type)

        if args.version_id != -1:  # leave-one-out
            xfund_dir = base_data_dir
    
    # xfund_dir = os.path.join(base_data_dir, 'Eval/temp')
    # xfund_dir = os.path.join(base_data_dir, version_name)
    
    if args.data_range == -1:
        data_range = range(len(anno_names))
        # data_range = range(len(ocr_names) -1, len(ocr_names)+1)
    else:
        data_range = range(args.data_range)
    
    documents = []
    print(xfund_dir)
    for i in data_range:
        print(i, img_names[i], anno_names[i])
        
        anno_file = os.path.join(anno_dir, anno_names[i])
        with open(anno_file, 'r', encoding='utf-8') as json_f:
            anno = json.load(json_f)
            form = anno['form']
        
        documents.append(form)
    
    if xfund_dir != None:
        if args.data_type == 'train':
            write_xfund_classificatino(documents, img_dir=img_dir, save_dir=xfund_dir, error=True)
        else:
            write_xfund_classificatino(documents, img_dir=img_dir, save_dir=xfund_dir)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ESun')
    parser.add_argument('--version_id', type=int, default=-1)
    parser.add_argument('--data_type', type=str, default='test')             # train, test

    parser.add_argument('--data_range', type=int, default=-1)  # -1 indicate all examples

    args = parser.parse_args()


    main(args)
