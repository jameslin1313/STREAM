import os
import json
import numpy as np
import argparse

import torch
from PIL import Image
from detectron2.data.detection_utils import read_image
from detectron2.data.transforms import ResizeTransform, TransformList


def normalize_bbox(bbox, w, h):
    # print(bbox, w, h)
    return [
        int(1000 * bbox[0] / w),
        int(1000 * bbox[1] / h),
        int(1000 * bbox[2] / w),
        int(1000 * bbox[3] / h),
    ]


def cell_construction(tokens, bboxes, width=10):  # width: width of cell
    left_points = [(row[0], i) for i, row in enumerate(bboxes)]  # left-x that define column
    top_points = [(row[1], i) for i, row in enumerate(bboxes)]   # left-y that define row
    
    cells = np.zeros((len(tokens), 2))  # (row, column)

    # define row using top coordinate
    row_id = 0
    top_sorted = sorted(top_points, key=lambda box:box[0])
    topmost = top_sorted[0][0]
    for point in top_sorted:
        y, i = point
        if y <= topmost + width:
            cells[i, 0] = row_id
        else:
            row_id += 1
            topmost = y
            cells[i, 0] = row_id
    
    # define column using left coordinate
    col_id = 0
    left_sorted = sorted(left_points, key=lambda box:box[0])
    leftmost = left_sorted[0][0]
    for point in left_sorted:
        x, i = point
        if x <= leftmost + width:
            cells[i, 1] = col_id
        else:
            col_id += 1
            leftmost = x
            cells[i, 1] = col_id
    
    return cells


def load_OCR(path, name=None, args=None):
    json_file = os.path.join(path, name)

    with open(json_file, encoding='utf-8') as json_f:
        anno = json.load(json_f)

        img_info = anno['img']
        w = img_info['width']
        h = img_info['height']

        document = anno['document']              # OCR
        label_entities = anno['label_entities']  # SER labels
        
        # load OCR results (char)
        charID_to_char = {}
        charID_to_charbox = {}
        charID_to_entityID = {}  # entityID is represent by the ID of first character
        
        for i, entity in enumerate(document):
            words = entity['words']

            # save total info. using character ID
            char_id = words[0]['id']

            # load char
            for word in words:
                c_id = word['id']
                # load char
                charID_to_char[c_id] = word['text']
                
                # load & convert bbox
                bndbox = word['bndbox']
                x1, x2 = min(row[0] for row in bndbox), max(row[0] for row in bndbox)
                y1, y2 = min(row[1] for row in bndbox), max(row[1] for row in bndbox)
                bbox = [x1, y1, x2, y2]
                
                charID_to_charbox[c_id] = bbox
                charID_to_entityID[c_id] = char_id

        # load groundtruth grouping & SER
        grouping_entities = []
        grouping_id = 0
        used_ids = []
        unused_ids = []

        for _, entity_label in enumerate(label_entities):
            linking_id = entity_label['entity_id']
            label = entity_label['label']
            word_idxs = sorted(entity_label['word_idx'])
            
            text = ''
            bndbox = []
            ids = []
            for word_id in word_idxs:
                text += charID_to_char[word_id]
                bndbox.append(charID_to_charbox[word_id])
                ids.append(word_id)
                used_ids.append(word_id)
            
            x1, x2 = min(row[0] for row in bndbox), max(row[2] for row in bndbox)
            y1, y2 = min(row[1] for row in bndbox), max(row[3] for row in bndbox)
            
            grouping_entity = {}
            grouping_entity['id'] = ids[0]
            grouping_entity['text'] = text
            grouping_entity['box'] = [x1, y1, x2, y2]
            grouping_entity['label'] = label
            grouping_entity['linking_id'] = linking_id
            grouping_entities.append(grouping_entity)
        
        # determine unused words as others
        for i in range(len(charID_to_char)):
            if i not in used_ids:
                unused_ids.append(i)

        remained_text = ''
        pre_entity_id = -1
        bndbox = []

        for unused_id in unused_ids:
            if charID_to_entityID[unused_id] != pre_entity_id and pre_entity_id != -1 and remained_text.split() != []:
                remained_entity = {}
                if remained_text == ' ':
                    print((remained_text != '' or remained_text != ' '))
                remained_entity['id'] = pre_entity_id  # word id
                remained_entity['text'] = remained_text
                x1, x2 = min(row[0] for row in bndbox), max(row[2] for row in bndbox)
                y1, y2 = min(row[1] for row in bndbox), max(row[3] for row in bndbox)
                remained_entity['box'] = [x1,y1,x2,y2]
                remained_entity['label'] = 'other'
                remained_entity['linking_id'] = -1

                # add entity to total entities
                grouping_entities.append(remained_entity)
                
                # reset
                remained_text = ''
                bndbox = []
            
            # accumulate entity
            remained_text += charID_to_char[unused_id]
            bndbox.append(charID_to_charbox[unused_id])
            pre_entity_id = charID_to_entityID[unused_id]
        
        # append last entity
        if remained_text.split() != []:
            remained_entity = {}
            remained_entity['id'] = pre_entity_id  # word id
            remained_entity['text'] = remained_text
            x1, x2 = min(row[0] for row in bndbox), max(row[2] for row in bndbox)
            y1, y2 = min(row[1] for row in bndbox), max(row[3] for row in bndbox)
            remained_entity['box'] = [x1,y1,x2,y2]
            remained_entity['label'] = 'other'
            remained_entity['linking_id'] = -1

            grouping_entities.append(remained_entity)

        # construct grouping ground truth text inputs
        grouping_entities = sorted(grouping_entities, key=lambda x: x['id'])

        # entity -> tokens
        tokens = []
        original_boxes = []
        bboxes = []
        world_ids = []     # [start char id, end char id, true grouing entity id]
        grouping_ser = []  # grouping_ser: (entity_id, ser_label)
        count = 0

        for entity_id, entity in enumerate(grouping_entities):
            ID = entity['id']
            text = entity['text']
            box = entity['box']
            label = entity['label']

            # print(entity)
            # print(text, box, label)
        
            # entity -> tokens
            words = text.split()

            # words -> bbox
            split_indices = []
            start_index = 0

            for word in words:
                index = text.find(word, start_index)
                split_indices.append(index)
                start_index = index + len(word)
            
            start_world_id = ID
            if len(split_indices) == 0:  # blank space, pass
                continue
            
            elif len(split_indices) == 1:  # single words
                tokens.append(words[0])
                original_boxes.append(box)
                box = normalize_bbox(box, w, h)
                bboxes.append(box)
                # print(words[0], box)

                world_id = [start_world_id, start_world_id + len(text) - 1, entity_id]
                world_ids.append(world_id)
                
                grouping_ser.append((entity_id, label))
                count += 1
            
            else:  # multiple words
                for i in range(0, len(split_indices)-1):
                    tokens.append(text[split_indices[i]:split_indices[i+1]-1])
                    start_id = split_indices[i] + start_world_id
                    end_id = split_indices[i+1] - 2 + start_world_id

                    start_box = charID_to_charbox[start_id]
                    end_box = charID_to_charbox[end_id]

                    bbox = [start_box[0], start_box[1], end_box[2], end_box[3]]
                    original_boxes.append(bbox)
                    bbox = normalize_bbox(bbox, w, h)
                    bboxes.append(bbox)
                    # print(text[split_indices[i]:split_indices[i+1]-1], bbox, start_id, end_id)

                    world_id = [start_world_id + start_id, start_world_id + end_id, entity_id]
                    world_ids.append(world_id)
                    
                    grouping_ser.append((entity_id, label))
                    count += 1
                
                # append last word
                tokens.append(text[split_indices[len(split_indices)-1]:])

                start_id = split_indices[len(split_indices)-1] + start_world_id
                end_id = len(text) - 1 + start_world_id

                start_box = charID_to_charbox[start_id]
                end_box = charID_to_charbox[end_id]

                bbox = [start_box[0], start_box[1], end_box[2], end_box[3]]
                original_boxes.append(bbox)
                bbox = normalize_bbox(bbox, w, h)
                bboxes.append(bbox)
                # print(text[split_indices[len(split_indices)-1]:], bbox)

                world_id = [start_world_id + start_id, start_world_id + end_id, entity_id]
                world_ids.append(world_id)

                grouping_ser.append((entity_id, label))
                count += 1
        
        if args.sort_type == 'box':
            combined_data = list(zip(tokens, bboxes, original_boxes, grouping_ser))  # Combine text and coordinates into tuples
            sorted_data = sorted(combined_data, key=lambda x: (x[1][1], x[1][0]))  # Sort the combined data based on the top-left corner coordinates (y1, x1)
            tokens, bboxes, original_boxes, grouping_ser = zip(*sorted_data)  # Unpack sorted data into separate lists
        elif args.sort_type == 'cell':
            cells = cell_construction(tokens, bboxes)  # define cell
            combined_data = list(zip(tokens, bboxes, original_boxes, grouping_ser, cells))  # Combine text and coordinates into tuples
            sorted_data = sorted(combined_data, key=lambda x: (x[4][0], x[4][1], x[1][1], x[1][0]))  # Sort the combined data based on the top-left corner coordinates (row, col, y1, x1)
            tokens, bboxes, original_boxes, grouping_ser, cells = zip(*sorted_data)  # Unpack sorted data into separate lists
        elif args.sort_type == 'no':
            pass
        else:
            print('Error sort_type, should be "no" / "box" / "cell"')
        
        # for i in range(len(tokens)):
        #     print(cells[i], tokens[i], bboxes[i], grouping_ser[i])
        
        image_path = path + '/images/' + name[6:-5] + '.png'
        information = {'tokens': tokens, 'bboxes': bboxes, 'original_boxes': original_boxes,'labels': grouping_ser, 'linkings': None, 'image_path': image_path}
        
        if args.save_json:
            print(name[6:])
            save_dir = os.path.join(os.getcwd(), 'CORD-r/' + args.save_dir + '/' + args.data_type)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            with open(os.path.join(save_dir, name[6:]), 'w') as f:
                json.dump(information, f)

        return tokens, bboxes, len(grouping_entities), grouping_ser
    
    
def main(args):  # load CORD-r
    base_data_path = '/media/ai2lab/4TB SSD/Datasets/Token-Path-Prediction-Datasets/CORD-r'
    data_list_name = 'data.' + args.data_type + '.txt'
    data_list = os.path.join(base_data_path, data_list_name)

    img_name = []
    anno_name = []
    with open(data_list, 'r') as f:
        for line in f.readlines():
            img, anno = line.split()
            
            img_name.append(img)
            anno_name.append(anno)

    img_name = sorted(img_name)
    anno_name = sorted(anno_name)

    if args.data_range == -1:
        data_range = range(len(anno_name))
    else:
        data_range = range(args.data_range)
    
    transformed_data = []
    print(len(anno_name))

    for i in data_range:
        print(i, img_name[i])
        
        # anno_name[i] = 'jsons/train_0674.json'
        tokens, boxes, ent_num, grouping_ser = load_OCR(base_data_path, anno_name[i], args)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, default='train')
    parser.add_argument('--data_range', type=int, default=-1)  # -1 indicate all examples
    parser.add_argument('--save_json', action='store_false')
    parser.add_argument('--save_dir', type=str, default='cell-sorted')
    parser.add_argument('--sort_type', type=str, default='cell')  # type: [no, box, cell]
    args = parser.parse_args()
    
    main(args)