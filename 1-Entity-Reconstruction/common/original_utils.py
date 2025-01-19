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
    

    def tokenize_and_align_labels(self, examples, ent2id, augmentation=False, visual_embed=True, label_all_tokens=False, max_seq_len=512, data_type='train', img_id=0, exp_name='FUNSD-r'):
        # print(len(examples['tokens']))
        # print(examples['tokens'])

        # max_seq_len is set to 3072 for large document
        if exp_name == 'FUNSD-r' or exp_name == 'CORD-r':
        # if True:
            # English, microsoft/layoutlmv3-base
            # tokenized_inputs, dict_keys(['input_ids', 'attention_mask', 'overflow_to_sample_mapping', 'bbox'])
            tokenized_inputs = self.tokenizer(
                text = examples["tokens"],
                boxes = examples["bboxes"],
                truncation=True,
                padding=True,
                # padding=False,
                return_overflowing_tokens=True,
                # return_tensors='pt',
                # We use this argument because the texts in our dataset are lists of words (with a label for each word).
                # is_split_into_words=True,
            )
        else:
            # Chinese, microsoft/layoutlmv3-base-chinese
            # tokenized_inputs, dict_keys(['input_ids', 'attention_mask', 'overflow_to_sample_mapping'])
            tokenized_inputs = self.tokenizer(
                text = examples["tokens"],
                boxes = examples["bboxes"],
                truncation=True,
                # padding='max_length',
                padding=False,
                return_overflowing_tokens=True,
                # return_tensors='pt',
                is_split_into_words=True
            )
        
        # print(len(input_ids))
        # bbox = tokenized_inputs['bbox'].numpy()
        # print(bbox.shape)

        # tokenized_inputs, dict_keys(['input_ids', 'attention_mask', 'overflow_to_sample_mapping'])
        word_ids_size = len(tokenized_inputs["input_ids"])  # need this to flatten & append word_ids
        
        # flatten (510, 511 are special tokens)
        # input_ids = [element[:510] for sublist in tokenized_inputs['input_ids'] for element in sublist]
        # attention_mask = [element[:510] for sublist in tokenized_inputs['attention_mask'] for element in sublist]
        
        input_ids_list = []
        attention_mask_list = []
        word_ids_list = []
        # flatten (Note that 510-th, 511-th elements are special tokens !!)
        for batch_index in range(word_ids_size):
            if batch_index == 0:
                start_concate_index = 0
                max_concate_index = 510
            elif batch_index != (word_ids_size - 1):
                start_concate_index = 1
                max_concate_index = 510  # (Note that 510-th, 511-th elements are special tokens !!)
            else:
                start_concate_index = 1
                max_concate_index = 512

            single_input_ids = tokenized_inputs['input_ids'][batch_index][start_concate_index:max_concate_index]
            single_attention_mask = tokenized_inputs['attention_mask'][batch_index][start_concate_index:max_concate_index]
            single_word_ids = tokenized_inputs.word_ids(batch_index=batch_index)[start_concate_index:max_concate_index]  # token -> word ID
            
            input_ids_list.append(single_input_ids)
            attention_mask_list.append(single_attention_mask)
            word_ids_list.append(single_word_ids)
            # org_batch_index = tokenized_inputs["overflow_to_sample_mapping"][batch_index]  # document ID, in this usage is always 0
        
        input_ids = [element for sublist in input_ids_list for element in sublist]
        attention_mask = [element for sublist in attention_mask_list for element in sublist]
        word_ids = [element for sublist in word_ids_list for element in sublist]
        # print('-------------------------')
        # print(word_ids)
        # print(len(word_ids))
        # print(stop)
        # append to max_seq_len
        pad_num = max_seq_len - len(input_ids)
        if pad_num < 0:
            print('Need Larger Size !!')
            exit(-1)
        
        for i in range(pad_num):
            input_ids.append(1)       # pad 1
            attention_mask.append(0)  # pad 0
            word_ids.append(None)
        
        tokenized_inputs['input_ids'] = input_ids
        tokenized_inputs['attention_mask'] = attention_mask
        print(len(word_ids))
        # print(stop)

        # variables for labels (TPP matrix)
        ent_type_size = len(ent2id)
        labels = None  # (entity_type, input_d, input_d)
        paths = {}
        entity_ser = {}

        if data_type != "predict":
            labels = np.zeros((ent_type_size, max_seq_len, max_seq_len))
        
        bboxes = []
        images = []
        max_input_ent_idx = -1

        bbox = examples["bboxes"]
        bbox_inputs = []

        if data_type != "predict":
            # convert labels
            # ent2id_list = [ent[0] for ent in examples['ner_tags']]
            if max_input_ent_idx == -1:  # not truncated
                max_input_ent_idx = max([ent[0] for ent in examples['ner_tags']]) + 1
            # print(max_input_ent_idx)
            # initial path
            for i in range(max_input_ent_idx):
                paths[i] = []

            for i, word_idx in enumerate(word_ids):
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    bbox_inputs.append([0, 0, 0, 0])
                else:
                    # add to grouping path
                    ent_id = examples['ner_tags'][word_idx][0]
                    ent_type = examples['ner_tags'][word_idx][1]
                    # if ent_type != 'other':
                    paths[ent_id].append(i)
                    entity_ser[ent_id] = ent_type
                    bbox_inputs.append(bbox[word_idx])
            
            # convert path to GP labels
            for entid in paths.keys():
                pre_id = -1

                if len(paths[entid]) == 1:  # single words
                    idx = paths[entid][0]
                    labels[0, idx, idx] = 1
                    # print('single, ', idx)
                else:
                    for i, idx in enumerate(paths[entid]):
                        if pre_id != -1:
                            labels[0, pre_id, idx] = 1
                            # print(pre_id, idx)
                    
                        pre_id = idx
                    # labels[0, pre_id, pre_id] = 1  # last token links to itself
            # remove empty path
            for i in range(max_input_ent_idx):
                if len(paths[i]) == 0:
                    del paths[i]
            
        # inference, don't need labels
        else:
            for i, word_idx in enumerate(word_ids):
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    bbox_inputs.append([0, 0, 0, 0])
                else:
                    bbox_inputs.append(bbox[word_idx])

        # bboxes.append(bbox_inputs)
            
        if visual_embed:
            # ipath = examples["image_path"][org_batch_index]
            ipath = examples["image_path"]
            img = self.pil_loader(ipath)
            for_patches, _ = self.common_transform(img, augmentation=augmentation)
            patch = self.patch_transform(for_patches)
            images.append(patch)
        
        # tokenized_inputs["bbox"] = bboxes
        tokenized_inputs["bbox"] = bbox_inputs
        tokenized_inputs["labels"] = labels  # if data_type == 'predict', labels = None
        tokenized_inputs["paths"] = paths
        tokenized_inputs["entity_ser"] = entity_ser

        if visual_embed:
            # tokenized_inputs["images"] = images
            tokenized_inputs['images'] = [patch]  # (3,224,244)

        return tokenized_inputs

    
    def load_document(self, data_type='temp2'):
        documents = self.load_FUNSDr_document(data_type)

        return documents
    

    def load_FUNSDr_document_list(self, data_type='temp'):
        if self.exp_name == 'FUNSD-r':
            # data_dir = os.path.join(os.getcwd(), 'datasets/FUNSD-r/transformed_labels/' + data_type)
            data_dir = os.path.join(os.getcwd(), 'datasets/FUNSD-r/new_transformed/' + data_type)
        elif self.exp_name == 'CORD-r':
            # data_dir = os.path.join(os.getcwd(), 'datasets/CORD-r/transformed_labels/' + data_type)
            data_dir = os.path.join(os.getcwd(), 'datasets/CORD-r/cell-sorted/' + data_type)
        elif self.exp_name == 'ESun':
            data_dir = os.path.join(os.getcwd(), 'datasets/ESun/cell-sorted/' + data_type)
            # data_dir = os.path.join(os.getcwd(), 'datasets/ESun/cell-sorted/test')
        
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
    