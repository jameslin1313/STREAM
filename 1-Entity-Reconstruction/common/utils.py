"""
Date: 2021-06-01 22:29:43
LastEditors: GodK
LastEditTime: 2021-07-31 19:30:18
"""
import torch

import sys
import os
import json
import numpy as np
from .image_utils import load_image, normalize_bbox
from common.image_utils import RandomResizedCropAndInterpolationWithTwoPic, pil_loader, Compose

from PIL import Image
from torchvision import transforms
from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
import shutil


def multilabel_categorical_crossentropy(y_true, y_pred):
    """
    https://kexue.fm/archives/7359
    """
    y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
    y_pred_pos = (y_pred - (1 - y_true) * 1e12)  # mask the pred outputs of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

    return (neg_loss + pos_loss).mean()


class Preprocessor(object):
    def __init__(self, tokenizer, add_special_tokens=True):
        super(Preprocessor, self).__init__()
        self.tokenizer = tokenizer
        self.add_special_tokens = add_special_tokens

    def get_ent2token_spans(self, text, entity_list):
        """实体列表转为token_spans

        Args:
            text (str): 原始文本
            entity_list (list): [(start, end, ent_type),(start, end, ent_type)...]
        """
        ent2token_spans = []

        inputs = self.tokenizer(text, add_special_tokens=self.add_special_tokens, return_offsets_mapping=True)
        token2char_span_mapping = inputs["offset_mapping"]
        text2tokens = self.tokenizer.tokenize(text, add_special_tokens=self.add_special_tokens)

        for ent_span in entity_list:  # ent_span: (start, end, label)
            ent = text[ent_span[0]:ent_span[1] + 1]
            ent2token = self.tokenizer.tokenize(ent, add_special_tokens=False)

            # 寻找ent的token_span
            token_start_indexs = [i for i, v in enumerate(text2tokens) if v == ent2token[0]]
            token_end_indexs = [i for i, v in enumerate(text2tokens) if v == ent2token[-1]]

            token_start_index = list(filter(lambda x: token2char_span_mapping[x][0] == ent_span[0], token_start_indexs))
            token_end_index = list(filter(lambda x: token2char_span_mapping[x][-1] - 1 == ent_span[1],
                                          token_end_indexs))  # token2char_span_mapping[x][-1]-1 减1是因为原始的char_span是闭区间，而token2char_span是开区间

            if len(token_start_index) == 0 or len(token_end_index) == 0:
                # print(f'[{ent}] 无法对应到 [{text}] 的token_span，已丢弃')
                continue
            token_span = (token_start_index[0], token_end_index[0], ent_span[2])
            ent2token_spans.append(token_span)
        
        return ent2token_spans


class DocumentLoader(object):
    def __init__(self, tokenizer, add_special_tokens=True, exp_name='FUNSD-r'):
        super().__init__()
        self.tokenizer = tokenizer
        self.add_special_tokens = add_special_tokens
        self.exp_name = exp_name

        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD
        self.common_transform = Compose([
            # transforms.ColorJitter(0.4, 0.4, 0.4),
            # transforms.RandomHorizontalFlip(p=0.5),
            RandomResizedCropAndInterpolationWithTwoPic(
                size=224, interpolation='bicubic'),
        ])

        self.patch_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])
    
    
    def pil_loader(self, path: str) -> Image.Image:
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    

    def load_FUNSDr_document(self, data_type='temp'):
        anno_files = os.listdir(data_dir)
        
        for guid, anno_file in enumerate(anno_files):
            json_file = os.path.join(data_dir, anno_file)
            with open(json_file, encoding='utf-8') as json_f:
                anno = json.load(json_f)
                tokens = anno['tokens']
                bboxes = anno['bboxes']
                labels = anno['labels']
                linkings = anno['linkings']
                image_path = anno['image_path']

            yield guid, {"id": str(guid), "tokens": tokens, "bboxes": bboxes, "ner_tags": labels, "linkings": linking, "image_path": image_path}
    

    # Chinese Tokenizer
    def tokenize_and_align_labels_chinese(self, examples, ent2id, augmentation=False, visual_embed=True, max_seq_len=512, data_type='train', img_id=0, exp_name='FUNSD-r', ignore_other=False, entity_grouped_label=False):
        # Chinese, microsoft/layoutlmv3-base-chinese
        tokenized_inputs = self.tokenizer(
            examples["tokens"],
            truncation=True,
            padding='max_length',
            return_overflowing_tokens=True,
            return_tensors='pt',
            is_split_into_words=True
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            # is_split_into_words=True,
        )

        ent_type_size = len(ent2id)
        entity_ser = {}
        
        bboxes = []
        images = []
        labels = []  # (entity_type, input_d, input_d)
        paths = []
        form = []

        for batch_index in range(len(tokenized_inputs["input_ids"])):
            word_ids = tokenized_inputs.word_ids(batch_index=batch_index)  # token -> word ID (list, len=512)
            org_batch_index = tokenized_inputs["overflow_to_sample_mapping"][batch_index]  # document ID, in this usage is always 0
            
            max_input_ent_idx = -1

            if tokenized_inputs['input_ids'].shape[0] >= 2:  # truncate tokens
                if data_type != "predict":
                    max_input_ent_idx = max([ent[0] for ent in examples['ner_tags'][:word_ids[511]]]) + 1  # truncate token > 512

            
            # bbox = examples["bboxes"][org_batch_index]
            bbox = examples["bboxes"]
            bbox_inputs = []
            label = None
            path = {}

            if data_type != "predict":
                label = np.zeros((ent_type_size, max_seq_len, max_seq_len))

            if data_type != "predict":
                # convert labels
                if max_input_ent_idx == -1:  # not truncated
                    max_input_ent_idx = max([ent[0] for ent in examples['ner_tags']]) + 1
                
                # initial path
                for i in range(max_input_ent_idx):
                    path[i] = []

                for i, word_idx in enumerate(word_ids):
                    # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                    # ignored in the loss function.
                    if word_idx is None:
                        bbox_inputs.append([0, 0, 0, 0])
                    else:
                        # add to grouping path
                        text = examples['tokens'][word_idx]
                        ent_id = examples['ner_tags'][word_idx][0]
                        ent_type = examples['ner_tags'][word_idx][1]
                        
                        path[ent_id].append(i)
                        entity_ser[ent_id] = ent_type
                        if text == '<BLANK>':
                            entity_ser[ent_id] = '<BLANK>'
                        bbox_inputs.append(bbox[word_idx])
                
                # remove empty path
                for i in range(max_input_ent_idx):
                    if len(path[i]) == 0:
                        del path[i]

                # convert path to GP labels
                # ignore_other = True
                for entid in path.keys():
                    pre_id = -1

                    ser_type = entity_ser[entid]
                    if ser_type == 'other' and ignore_other:
                        continue

                    if len(path[entid]) == 1:  # single words
                        idx = path[entid][0]
                        label[0, idx, idx] = 1
                    else:
                        for i, idx in enumerate(path[entid]):
                            if pre_id != -1:
                                label[0, pre_id, idx] = 1
                        
                            pre_id = idx
                    
                    # construct grouped label
                    if entity_grouped_label:
                        if len(path[entid]) == 1:  # single words
                            idx = path[entid][0]
                            label[1, idx, idx] = 1
                        else:
                            idx_list = path[entid]
                            for i, current_idx in enumerate(path[entid]):
                                for idx in idx_list:
                                    label[1, current_idx, idx] = 1
                        
                
            # inference, don't need labels
            else:
                for i, word_idx in enumerate(word_ids):
                    # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                    # ignored in the loss function.
                    if word_idx is None:
                        bbox_inputs.append([0, 0, 0, 0])
                    else:
                        bbox_inputs.append(bbox[word_idx])

            bboxes.append(bbox_inputs)
            labels.append(label)
            paths.append(path)



            decode = path
            for id, ent_id in enumerate(decode.keys()):
                token_list = decode[ent_id]
                
                token_ids = []
                for i in token_list:
                    token_ids.append(word_ids[i])
                token_ids = list(set(token_ids))
                
                # collect OCR information
                v_text = ''
                v_tokens = []
                v_boxes = []
                v_original_boxes = []
                v_labels = []
                for i in token_ids:
                    if i == None:
                        continue
                    v_text += examples['tokens'][i] + ' '
                    v_tokens.append(examples['tokens'][i])
                    v_boxes.append(examples['bboxes'][i])
                    v_original_boxes.append(examples['original_boxes'][i])
                    v_labels.append(examples['ner_tags'][i][1])
                
                # handel exception
                if len(v_boxes) == 0 or len(v_tokens) == 0:
                    continue
                
                # FUNSD format
                v_words = []
                for i in range(len(v_tokens)):
                    v_word = {"text": v_tokens[i], "box": v_original_boxes[i]}
                    v_words.append(v_word)

                # transform to entity-level box & labels
                x1, x2 = min(row[0] for row in v_original_boxes), max(row[2] for row in v_original_boxes)
                y1, y2 = min(row[1] for row in v_original_boxes), max(row[3] for row in v_original_boxes)
                v_box = [x1, y1, x2, y2]
                
                v_label = v_labels[0]
                if v_text == '<BLANK> ':  # space is essential
                    v_label = '<BLANK>'

                ocr = {"text": v_text, "box": v_box, "label": v_label, "words": v_words, "id": id}
                
                form.append(ocr)

        prediction = {"form": form}

        # label visualization
        # if exp_name == 'XFUND-r':
        #     if data_type == 'train':
        #         start_id = 69  # 76 for FUNSD / 78 for FUNSD-r
        #     else:
        #         start_id = 71  # 75 for FUNSD / 77 for FUNSD-r
        # elif exp_name == 'ESun':
        #     if data_type == 'train':
        #         start_id = 69  # ESun: 66 / ESun-v2: 69
        #     else:
        #         start_id = 68  # ESun: 65 / ESun-v2: 68
        # elif exp_name == 'Synthetic-form':
        #     if data_type == 'train':
        #         start_id = 79
        #     else:
        #         start_id = 78
        # else:
        #     # combined
        #     if examples['image_path'][66:-4].isnumeric():
        #         if data_type == 'train':
        #             start_id = 66  # XFUND: 69 / ESun: 
        #         else:
        #             start_id = 65  # XFUND: 68 / ESun: 
        #     else:
        #         if data_type == 'train':
        #             start_id = 79  # XFUND: 76 / ESun: 
        #         else:
        #             start_id = 78  # XFUND: 75 / ESun: 
        
        # filename = examples['image_path'][start_id:-4] + '.json'  # start_id is path dependent
        # save_dir = 'visualize/' + exp_name + '-visualize/' + data_type

        # if img_id == 0:
        #     if os.path.exists(save_dir):
        #         shutil.rmtree(save_dir)  # If it exists, remove it along with its contents
        #     os.makedirs(save_dir)

        # save_file = os.path.join(save_dir, filename)
        # with open(save_file, 'w', encoding='utf8') as f:
        #     print(img_id, save_file)
        #     json.dump(prediction, f, ensure_ascii=False, indent=4)

        
        if visual_embed:
            ipath = examples["image_path"]
            img = self.pil_loader(ipath)
            for_patches, _ = self.common_transform(img, augmentation=augmentation)
            patch = self.patch_transform(for_patches)  # patch: [3,224,224]
        
        tokenized_inputs["bbox"] = bboxes
        tokenized_inputs["labels"] = np.array(labels)  # if data_type == 'predict', labels = None
        tokenized_inputs["paths"] = paths
        tokenized_inputs["entity_ser"] = entity_ser

        print('Document Shape: ', tokenized_inputs['labels'].shape)

        if visual_embed:
            tokenized_inputs["images"] = [patch]

        return tokenized_inputs


    # English tokenizer
    def tokenize_and_align_labels(self, examples, ent2id, augmentation=False, visual_embed=True, label_all_tokens=False, max_seq_len=512, data_type='train', img_id=0, exp_name='FUNSD-r', ignore_other=False, entity_grouped_label=False):
        ignore_other = True
        # English, microsoft/layoutlmv3-base
        tokenized_inputs = self.tokenizer(
            text = examples["tokens"],
            boxes = examples["bboxes"],
            truncation='only_first',
            padding='max_length',
            return_overflowing_tokens=True,
            return_tensors='pt'
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            # is_split_into_words=True,
        )
        
        ent_type_size = len(ent2id)
        entity_ser = {}
        
        bboxes = []
        images = []
        labels = []  # (entity_type, input_d, input_d)
        paths = []
        form = []

        for batch_index in range(len(tokenized_inputs["input_ids"])):
            word_ids = tokenized_inputs.word_ids(batch_index=batch_index)  # token -> word ID (list, len=512)
            org_batch_index = tokenized_inputs["overflow_to_sample_mapping"][batch_index]  # document ID, in this usage is always 0
            
            max_input_ent_idx = -1

            if tokenized_inputs['input_ids'].shape[0] >= 2:  # truncate tokens
                if data_type != "predict":
                    max_input_ent_idx = max([ent[0] for ent in examples['ner_tags'][:word_ids[511]]]) + 1  # truncate token > 512

            
            # bbox = examples["bboxes"][org_batch_index]
            bbox = examples["bboxes"]
            bbox_inputs = []
            label = None
            path = {}

            if data_type != "predict":
                # label initialization
                label = np.zeros((ent_type_size, max_seq_len, max_seq_len))
                
                # convert labels
                if max_input_ent_idx == -1:  # not truncated
                    max_input_ent_idx = max([ent[0] for ent in examples['ner_tags']]) + 1
                
                # initial path
                for i in range(max_input_ent_idx):
                    path[i] = []

                for i, word_idx in enumerate(word_ids):
                    # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                    # ignored in the loss function.
                    if word_idx is None:
                        bbox_inputs.append([0, 0, 0, 0])
                    else:
                        # add to grouping path
                        ent_id = examples['ner_tags'][word_idx][0]
                        ent_type = examples['ner_tags'][word_idx][1]
                        
                        path[ent_id].append(i)
                        entity_ser[ent_id] = ent_type
                        bbox_inputs.append(bbox[word_idx])
                
                # remove empty path
                for i in range(max_input_ent_idx):
                    if len(path[i]) == 0:
                        del path[i]
                             
                # convert path to GP labels
                for entid in path.keys():
                    pre_id = -1

                    ser_type = entity_ser[entid]
                    if ser_type == 'other' and ignore_other:
                        continue

                    if len(path[entid]) == 1:  # single words
                        idx = path[entid][0]
                        label[0, idx, idx] = 1
                    else:
                        for i, idx in enumerate(path[entid]):
                            if pre_id != -1:
                                label[0, pre_id, idx] = 1
                        
                            pre_id = idx
                    
                    # construct grouped label
                    if entity_grouped_label:
                        if len(path[entid]) == 1:  # single words
                            idx = path[entid][0]
                            label[1, idx, idx] = 1
                        else:
                            idx_list = path[entid]
                            for i, current_idx in enumerate(path[entid]):
                                for idx in idx_list:
                                    label[1, current_idx, idx] = 1

                
            # inference, don't need labels
            else:
                for i, word_idx in enumerate(word_ids):
                    # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                    # ignored in the loss function.
                    if word_idx is None:
                        bbox_inputs.append([0, 0, 0, 0])
                    else:
                        bbox_inputs.append(bbox[word_idx])

            bboxes.append(bbox_inputs)
            labels.append(label)
            paths.append(path)


            decode = path
            for id, ent_id in enumerate(decode.keys()):
                token_list = decode[ent_id]
                
                token_ids = []
                for i in token_list:
                    token_ids.append(word_ids[i])
                token_ids = list(set(token_ids))
                
                # collect OCR information
                v_text = ''
                v_tokens = []
                v_boxes = []
                v_original_boxes = []
                v_labels = []
                for i in token_ids:
                    if i == None:
                        continue
                    v_text += examples['tokens'][i] + ' '
                    v_tokens.append(examples['tokens'][i])
                    v_boxes.append(examples['bboxes'][i])
                    v_original_boxes.append(examples['original_boxes'][i])
                    v_labels.append(examples['ner_tags'][i][1])
                
                # handel exception
                if len(v_boxes) == 0 or len(v_tokens) == 0:
                    continue
                
                if exp_name == 'FUNSD-r' or exp_name == 'EC-FUNSD-r':
                    # FUNSD format
                    v_words = []
                    for i in range(len(v_tokens)):
                        v_word = {"text": v_tokens[i], "box": v_original_boxes[i]}
                        v_words.append(v_word)

                    # transform to entity-level box & labels
                    x1, x2 = min(row[0] for row in v_original_boxes), max(row[2] for row in v_original_boxes)
                    y1, y2 = min(row[1] for row in v_original_boxes), max(row[3] for row in v_original_boxes)
                    v_box = [x1, y1, x2, y2]
                    
                    v_label = v_labels[0]

                    ocr = {"text": v_text, "box": v_box, "label": v_label, "words": v_words, "id": id}
                    
                    form.append(ocr)

        prediction = {"form": form}
        
        # Label visualization
        # if data_type == 'train':
        #     start_id = 76  # 76 for FUNSD / 78 for FUNSD-r
        # else:
        #     start_id = 75  # 75 for FUNSD / 77 for FUNSD-r
        # filename = examples['image_path'][start_id:-4] + '.json'  # start_id is path dependent
        # save_dir = 'visualize/FUNSD-r-visualize/' + data_type

        # if img_id == 0:
        #     if os.path.exists(save_dir):
        #         shutil.rmtree(save_dir)  # If it exists, remove it along with its contents
        #     os.makedirs(save_dir)

        # # print(filename)
        # save_file = os.path.join(save_dir, filename)
        # with open(save_file, 'w', encoding='utf8') as f:
        #     print(img_id, save_file)
        #     json.dump(prediction, f, ensure_ascii=False, indent=4)


        if visual_embed:
            ipath = examples["image_path"]
            img = self.pil_loader(ipath)
            for_patches, _ = self.common_transform(img, augmentation=augmentation)
            patch = self.patch_transform(for_patches)
        
        tokenized_inputs["bbox"] = bboxes
        tokenized_inputs["labels"] = np.array(labels)  # if data_type == 'predict', labels = None
        tokenized_inputs["paths"] = paths
        tokenized_inputs["entity_ser"] = entity_ser

        print(tokenized_inputs['labels'].shape)

        if visual_embed:
            tokenized_inputs["images"] = [patch]

        return tokenized_inputs

    

    def load_FUNSDr_document_list(self, data_type='temp'):
        if self.exp_name == 'ESun':
            data_dir = os.path.join(os.getcwd(), 'datasets/ESun-v2/cell-sorted/' + data_type)
        elif self.exp_name == 'Combined':
            data_dir = os.path.join(os.getcwd(), 'datasets/ESun_and_Synthetic/not-sorted/' + data_type)
        elif self.exp_name == 'leave-one-out':
            version = "setting_00/"
            data_dir = os.path.join(os.getcwd(), 'datasets/leave-one-out/not-sorted/' + version + data_type)
        else:
            data_dir = os.path.join(os.getcwd(), f"datasets/{self.exp_name}/not-sorted/{data_type}")
        
        anno_files = sorted(os.listdir(data_dir))
        full_labels = []
        for guid, anno_file in enumerate(anno_files):
            json_file = os.path.join(data_dir, anno_file)
            with open(json_file, encoding='utf-8') as json_f:
                anno = json.load(json_f)

                tokens = anno['tokens']
                bboxes = anno['bboxes']
                original_boxes = anno['original_boxes']
                # original_boxes = bboxes
                labels = anno['labels']
                linkings = anno['linkings']
                # linkings = None
                image_path = anno['image_path']

            label = (guid, {"id": str(guid), "tokens": tokens, "bboxes": bboxes, "original_boxes": original_boxes, "ner_tags": labels, "linkings": linkings, "image_path": image_path})
            full_labels.append(label)
        
        return full_labels
    
    def load_document_list(self, data_type='temp2'):
        documents = self.load_FUNSDr_document_list(data_type)

        return documents
    