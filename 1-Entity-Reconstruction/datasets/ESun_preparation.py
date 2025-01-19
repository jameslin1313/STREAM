import os
import json
import numpy as np
import argparse
import cv2
import shutil


def main(args):
    # load dataset (document images & annotations)
    if args.dataset != 'TransGlobe-r':
        base_data_path = '/media/ai2lab/4TB SSD/Datasets/' + args.dataset + '/grouping/' + args.data_type + 'ing_data'
        img_dir = os.path.join(base_data_path, 'images')
        anno_dir = os.path.join(base_data_path, 'annotations')
    elif args.dataset == 'TransGlobe-r':
        # data_type: Medical, Medical-Receipt
        if args.data_type == 'Medical' or args.data_type == 'Medical-Receipt':
            img_dir = '/media/ai2lab/4TB SSD/Datasets/TransGlobe/' + args.data_type + '/images'
            anno_dir = '/media/ai2lab/4TB SSD/Datasets/TransGlobe/' + args.data_type + '/OCRs'

        # anno mode
        elif args.data_type == 'test' and args.version_id == -1:
            img_dir = '/media/ai2lab/4TB SSD/Datasets/TransGlobe/Dataset-filled/testing_data/images'
            anno_dir = '/media/ai2lab/4TB SSD/Datasets/TransGlobe-r/Azure-OCR/grouping/testing_data/annotations'

        # leave-one-out mode
        elif args.version_id != -1:
            img_dir = os.path.join('/media/ai2lab/4TB SSD/Datasets/leave-one-out/Azure-OCR/leave-one-out_grouping', f"setting_{args.version_id:02d}", args.data_type + 'ing_data/images')
            anno_dir = os.path.join('/media/ai2lab/4TB SSD/Datasets/leave-one-out/Azure-OCR/leave-one-out_grouping', f"setting_{args.version_id:02d}", args.data_type + 'ing_data/annotations')
        else:
            print('Data Type of TransGlobe Should Be "Medical" or "Medical-Receipt" !')
            exit(-1)

    img_name = sorted(os.listdir(img_dir))
    anno_name = sorted(os.listdir(anno_dir))


    visualize_save_dir = '/home/ai2lab/Desktop/VDU/LayoutLM_GP/visualize/' + args.dataset + '-visualize/' + args.data_type
    if os.path.exists(visualize_save_dir):
        shutil.rmtree(visualize_save_dir)  # If it exists, remove it along with its contents
    os.makedirs(visualize_save_dir)


    if args.data_range == -1:
        data_range = range(len(anno_name))
    else:
        data_range = range(args.data_range)

    for i in data_range:
        print(i, img_name[i])
        
        # get image size
        img_file = os.path.join(img_dir, img_name[i])
        anno_file = os.path.join(anno_dir, anno_name[i])

        # construct grouping label
        # write_layoutlm_gp(img_file, anno_file, args=args)
        with open(anno_file, encoding='utf-8') as json_f:
            anno = json.load(json_f)
            # print(anno.keys())

            tokens = anno['tokens']
            bboxes = anno['bboxes']
            original_boxes = anno['original_boxes']
            labels = anno['labels']
            linkings = anno['linkings']

            anno['image_path'] = img_file
            
            decode = {}
            for idx in range(len(tokens)):
                ent_id = labels[idx][0]
                if ent_id not in decode.keys():
                    decode[ent_id] = [idx]
                else:
                    decode[ent_id].append(idx)
            
            group_tokens = []
            group_bboxes = []
            group_original_boxes = []
            group_labels = []

            for ent_id in decode.keys():
                for idx in decode[ent_id]:
                    group_tokens.append(tokens[idx])
                    group_bboxes.append(bboxes[idx])
                    group_original_boxes.append(original_boxes[idx])
                    group_labels.append(labels[idx])
            
            group_anno = {'tokens': group_tokens, 'bboxes': group_bboxes, 'original_boxes': group_original_boxes, 'labels': group_labels, 'linkings': linkings, 'image_path': img_file}

            form = []
            for j in range(len(tokens)):
                form.append({'text': tokens[j], 'box': bboxes[j], 'label': labels[j][1]})
            funsd = {'form': form}

            # save FUNSD format for visualization
            filename = os.path.join(visualize_save_dir, anno_name[i])
            with open(filename, 'w') as f:
                json.dump(funsd, f, ensure_ascii=False, indent=4)
        


        if args.save_json:
            # check save dir
            if args.sort_type == 'cell':
                save_dir = os.path.join(os.getcwd(), args.dataset + '/cell-sorted/' + args.data_type)
                if args.version_id != -1:
                    save_dir = os.path.join(os.getcwd(), args.dataset + '/cell-sorted/' + f"setting_{args.version_id:02d}/" + args.data_type)
                
                write_anno = anno

            elif args.sort_type == 'no':
                save_dir = os.path.join(os.getcwd(), args.dataset + '/not-sorted/' + args.data_type)
                if args.version_id != -1:
                    save_dir = os.path.join(os.getcwd(), args.dataset + '/not-sorted/' + f"setting_{args.version_id:02d}/" + args.data_type)
                write_anno = group_anno
            
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            filename = os.path.join(save_dir, img_name[i][:-3] + 'json')
            
            # write file
            with open(filename, 'w', encoding='utf-8') as f:
                print(filename)
                json.dump(write_anno, f, ensure_ascii=False, indent=4)
            
            # write id mapping json
            source_file = 'XFUND-r/group2id_2.json'
            target_file = os.path.join(os.getcwd(), args.dataset, 'group2id_2.json')
            if not os.path.exists(target_file):
                shutil.copy(source_file, target_file)

            source_file = 'XFUND-r/grouping2id.json'
            target_file = os.path.join(os.getcwd(), args.dataset, 'grouping2id.json')
            if not os.path.exists(target_file):
                shutil.copy(source_file, target_file)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, default='train')
    parser.add_argument('--dataset', type=str, default='ESun')
    parser.add_argument('--data_range', type=int, default=-1)  # -1 indicate all examples
    parser.add_argument('--save_json', action='store_false')
    parser.add_argument('--sort_type', type=str, default='no')  # type: [no, box, cell]
    parser.add_argument('--version_id', type=int, default=-1)   # leave-one-out setting
    args = parser.parse_args()

    main(args)
