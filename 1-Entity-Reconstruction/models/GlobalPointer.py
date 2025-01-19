"""
Date: 2021-06-02 00:33:09
LastEditors: GodK
"""
import sys
import os
import json
from .image_utils import load_image, normalize_bbox

sys.path.append("../")
from common.utils import Preprocessor, DocumentLoader
from common.image_utils import RandomResizedCropAndInterpolationWithTwoPic, pil_loader, Compose
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

from PIL import Image
from torchvision import transforms
from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

import shutil
import re


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.length = len(data)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.length


class DataMaker(object):
    def __init__(self, tokenizer, add_special_tokens=True):
        super().__init__()
        self.tokenizer = tokenizer
        self.add_special_tokens = add_special_tokens
        self.preprocessor = Preprocessor(tokenizer, self.add_special_tokens)
        
        # self.documentloader = FUNSDDocumentLoader(tokenizer, self.add_special_tokens)
        self.documentloader = DocumentLoader(tokenizer, self.add_special_tokens)
    

    def generate_inputs(self, datas, max_seq_len, ent2id, data_type="train", use_image=False):
        """生成喂入模型的数据

        Args:
            datas (list): json格式的数据[{'text':'','entity_list':[(start,end,ent_type),()]}]
            max_seq_len (int): 句子最大token数量
            ent2id (dict): ent到id的映射
            data_type (str, optional): data类型. Defaults to "train".

        Returns:
            list: [(sample, input_ids, attention_mask, token_type_ids, labels),(),()...]
        """
        documents = self.documentloader.load_document(data_type=data_type)
        
        # original GP
        ent_type_size = len(ent2id)  # 实体类别

        all_inputs = []
        for sample in documents:
            # {'input_ids', 'attention_msk', 'overflow_to_sample_mapping', 'bbox', 'labels', 'images'}
            tokenized_inputs = self.documentloader.tokenize_and_align_labels(sample[1], ent2id, max_seq_len=max_seq_len)  # {'input_ids', 'attention_msk', 'overflow_to_sample_mapping', 'bbox', 'labels', 'images'}
            
            # "microsoft/layoutlmv3-base-chinese": dict_keys(['input_ids', 'attention_mask'])
            # BertTokenizerFast                  : dict_keys(['input_ids', 'token_type_ids', 'attention_mask'])

            input_ids = torch.tensor(tokenized_inputs["input_ids"]).long()
            attention_mask = torch.tensor(tokenized_inputs["attention_mask"]).long()
            bbox = torch.tensor(tokenized_inputs["bbox"])  # (N, 4) with top-left, right-bottom
            
            if tokenized_inputs['labels'] is not None:
                labels = torch.tensor(tokenized_inputs["labels"]).long()

            images = None
            if use_image:
                images = torch.unsqueeze(tokenized_inputs['images'][0], dim=0)

            # sample_input = (sample, input_ids, attention_mask, token_type_ids, labels)
            sample_input = (sample, input_ids, attention_mask, bbox, images, labels)

            all_inputs.append(sample_input)

        return all_inputs
    

    def generate_batch(self, batch_data, max_seq_len, ent2id, data_type="train"):
        index = 0  # special batch index

        batch_sample = batch_data[index][0]
        batch_input_ids = batch_data[index][1]
        batch_attention_mask = batch_data[index][2]
        batch_bbox = batch_data[index][3]
        batch_image = batch_data[index][4]
        
        batch_labels = batch_data[index][5] if data_type != "predict" else None
        
        batch_paths = batch_data[index][6] if data_type != "predict" else None
        batch_entity_ser = batch_data[index][7] if data_type != "predict" else None

        return batch_sample, batch_input_ids, batch_attention_mask, batch_bbox, batch_image, batch_labels, batch_paths, batch_entity_ser

    def decode_ent(self, pred_matrix):
        pass


class MetricsCalculator(object):
    def __init__(self):
        super().__init__()

    def get_sample_f1(self, y_pred, y_true):
        y_pred = torch.gt(y_pred, 0).float()
        return 2 * torch.sum(y_true * y_pred) / torch.sum(y_true + y_pred)

    def get_sample_precision(self, y_pred, y_true):
        y_pred = torch.gt(y_pred, 0).float()
        return torch.sum(y_pred[y_true == 1]) / (y_pred.sum() + 1)

    def get_evaluate_fpr(self, y_pred, y_true):
        y_pred = y_pred.cpu().numpy()
        y_true = y_true.cpu().numpy()
        pred = []
        true = []

        for b, l, start, end in zip(*np.where(y_pred > 0)):  # batch (=1), label, start_index, end_index
            pred.append((l, start, end))
        
        for l, start, end in zip(*np.where(y_true > 0)):
            true.append((l, start, end))

        R = set(pred)
        T = set(true)
        X = len(R & T)
        Y = len(R)
        Z = len(T)

        f1, precision, recall = 2 * X / (Y + Z+1e-6), X / (Y+1e-6), X / (Z+1e-6)
        return f1, precision, recall
    

    def get_entity_eval(self, decode, y_true, entity_ser, ignore_other=False, count=0):
        correct = 0
        ignored = 0  # ignore other type
        correct_ids = []

        for idx, ent in y_true.items():
            # if ignore_other and entity_ser[idx] == '<BLANK> ':
            if ignore_other and entity_ser[idx] == 'other':
                ignored += 1
                continue
            
            # not other
            find = False
            for ent_id, pred_ent in decode.items():
                if ent == pred_ent:
                    correct += 1
                    correct_ids.append(ent_id)
                    break

        if ignore_other:
            precision = correct / (len(decode) - ignored + 1e-6)
            recall = correct / (len(y_true) - ignored + 1e-6)
            f1 = 2 * precision * recall / (precision + recall + 1e-6)
        else:
            precision = correct / (len(decode) + 1e-6)
            recall = correct / len(y_true)
            f1 = 2 * precision * recall / (precision + recall + 1e-6)

        return precision, recall, f1, correct_ids


# CORD-r label convert
label_convert = {
    'menu.sub.nm': 'menu.sub_nm',
    'menu.sub.unitprice': 'menu.sub_unitprice',
    'menu.sub.cnt': 'menu.sub_cnt',
    'menu.sub.price': 'menu.sub_price',
    'menu.sub.etc': 'menu.sub_etc',
    'subtotal.subtotal.price': 'subtotal.subtotal_price',
    'subtotal.discount.price': 'subtotal.discount_price',
    'subtotal.service.price': 'subtotal.service_price',
    'subtotal.othersvc.price': 'subtotal.othersvc_price',
    'subtotal.tax.price': 'subtotal.tax_price',
    'total.total.price': 'total.total_price',
    'total.total.etc': 'total.total_etc',
    'total.menutype.cnt': 'total.menutype_cnt',
    'total.menuqty.cnt': 'total.menuqty_cnt'
}


class GlobalPointer(nn.Module):
    def __init__(self, encoder, ent_type_size, inner_dim, RoPE=True):
        super().__init__()
        self.encoder = encoder
        self.ent_type_size = ent_type_size
        self.inner_dim = inner_dim
        self.hidden_size = encoder.config.hidden_size
        self.dense = nn.Linear(self.hidden_size, self.ent_type_size * self.inner_dim * 2)

        self.RoPE = RoPE
        self.row_id = 0
        self.write = False

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        embeddings = embeddings.to(self.device)
        return embeddings

    # def forward(self, input_ids, attention_mask, token_type_ids):
    def forward(self, input_ids, bbox, attention_mask, image):
        self.device = input_ids.device

        # image = None
        if image == None:
            # print('no image')
            context_outputs = self.encoder(input_ids, bbox, attention_mask)
        else:
            # print('with image')
            context_outputs = self.encoder(input_ids, bbox, attention_mask, pixel_values=image)

        # last_hidden_state:(batch_size, seq_len, hidden_size)
        last_hidden_state = context_outputs[0]  # text only: (1,512,768) / with image: (1,709,768)

        if image != None:
            last_hidden_state = last_hidden_state[:,:512,:]

        batch_size = last_hidden_state.size()[0]
        seq_len = last_hidden_state.size()[1]

        # outputs:(batch_size, seq_len, ent_type_size*inner_dim*2)
        outputs = self.dense(last_hidden_state)  # text only: (1,512,128) / with image: (1,709,128)
        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)
        # outputs:(batch_size, seq_len, ent_type_size, inner_dim*2)
        outputs = torch.stack(outputs, dim=-2)
        # qw,kw:(batch_size, seq_len, ent_type_size, inner_dim)
        qw, kw = outputs[..., :self.inner_dim], outputs[..., self.inner_dim:]

        if self.RoPE:
            # pos_emb:(batch_size, seq_len, inner_dim)
            pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.inner_dim)
            # cos_pos,sin_pos: (batch_size, seq_len, 1, inner_dim)
            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
            qw2 = qw2.reshape(qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos

        # logits:(batch_size, ent_type_size, seq_len, seq_len)
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)

        # padding mask
        pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        # pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.ent_type_size, 512, 512)
        # pad_mask_h = attention_mask.unsqueeze(1).unsqueeze(-1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        # pad_mask = pad_mask_v&pad_mask_h
        logits = logits * pad_mask - (1 - pad_mask) * 1e12

        # 排除下三角
        # mask = torch.tril(torch.ones_like(logits), -1)
        # logits = logits - mask * 1e12

        # 排除ordering matrix的下三角
        
        first_tensor = logits[0, 0]
        mask = torch.tril(torch.ones_like(first_tensor), -1)  # Apply the lower triangular mask to the first tensor
        # print(first_tensor.shape, mask.shape)
        tril_first_tensor = first_tensor - mask * 1e12
        logits[0, 0] = tril_first_tensor  # Assign the modified tensor back to the original tensor
        

        return logits / self.inner_dim ** 0.5
    

    def decode(self, y_pred, groundtruth=False):  # matrix to grouping list
        y_pred = y_pred.cpu().numpy()
        pred = []

        for b, l, start, end in zip(*np.where(y_pred > 0)):  # batch (=1), label, start_index, end_index
        # for l, start, end in zip(*np.where(y_pred > 0)):  # batch (=1), label, start_index, end_index
            pred.append((l, start, end))
        
        # tranlate pred to grouping list
        decode = {}  # {ent_id: [id list]}
        ent_num = 0
        id2ent = {}
        for _, pre_id, next_id in pred:
            if pre_id not in id2ent.keys():
                decode[ent_num] = [pre_id, next_id]  # new entity
                id2ent[next_id] = ent_num
                ent_num += 1
            else:
                ent_id = id2ent[pre_id]
                
                ent_list = decode[ent_id]  # get current token list of the entity
                ent_list.append(next_id)   # add new token
                decode[ent_id] = ent_list  # update token list

                id2ent[next_id] = ent_id

        def remove_duplicates_preserve_order(lst):
            seen = set()
            return [x for x in lst if not (x in seen or seen.add(x))]
        
        # remove duplication (single word)
        final_decode = {}
        for idx, ent_ids in decode.items():
            final_decode[idx] = remove_duplicates_preserve_order(ent_ids)
        
        return final_decode


    def inference(self, decodes, tokenizer, batch_sample, exp_name='FUNSD-r', save_dir=None, correct_ids=None):
        # read texts of each token
        examples = batch_sample[1]  # examples: dict_keys(['id', 'tokens', 'bboxes', 'ner_tags', 'linking', 'image_path'])
        
        # English
        if exp_name == 'FUNSD-r' or exp_name == 'CORD-r' or exp_name == 'EC-FUNSD-r':
            tokenized_inputs = tokenizer(
                text = examples["tokens"],
                boxes = examples["bboxes"],
                truncation=True,
                padding='max_length',
                return_overflowing_tokens=True,
                return_tensors='pt'
            )
        # Chinese
        else:
            tokenized_inputs = tokenizer(
                examples["tokens"],
                truncation=True,
                padding='max_length',
                return_overflowing_tokens=True,
                return_tensors='pt',
                is_split_into_words=True
            )
        # input_token = tokenizer.tokenize(text)  # not work for LayoutLMv3 tokenizer

        # translate grouping list to OCR information
        # FUNSD-r format
        if exp_name != 'CORD-r':
            form = []
        
        # CORD-r format
        if exp_name == 'CORD-r':
            valid_line = []
            meta = {}
            roi = []

            # write meta info
            meta["version"] = "CORD-r"
            meta["split"] = exp_name
            meta["image_id"] = examples['id']
            img = load_image(examples['image_path'])
            meta["image_size"] = {"width": img[1][0], "height": img[1][1]}

            # write roi info
            min_x = min(row[0] for row in examples['bboxes'])
            min_y = min(row[1] for row in examples['bboxes'])
            max_x = max(row[2] for row in examples['bboxes'])
            max_y = max(row[3] for row in examples['bboxes'])
            box = [min_x, min_y, max_x, max_y]

            roi = {"x2": box[2], "y3": box[3], "x3": box[2], "y4": box[3], "x1": box[0], "y1": box[1], "x4": box[0], "y2": box[1]}
            # OCR (valid line) initialization
            grouping_words = {}
            grouping_word_boxes = {}
            grouping_labels = {}
        
        pattern = re.compile(r'[\u4e00-\u9fff]+')  # function for Chinese check

        # decode result
        word_ids = None
        for batch_index in range(len(tokenized_inputs["input_ids"])):
            word_ids = tokenized_inputs.word_ids(batch_index=batch_index)  # token -> word ID

            decode = decodes[batch_index]
            correct_idx = correct_ids[batch_index]

            for id, ent_id in enumerate(decode.keys()):
                token_list = decode[ent_id]
                
                token_ids = []
                for i in token_list:
                    token_ids.append(word_ids[i])
                token_ids = sorted(list(set(token_ids)))
                
                # collect OCR information
                text = ''
                tokens = []
                boxes = []
                original_boxes = []
                labels = []
                for i in token_ids:
                    if i == None:
                        continue

                    text += examples['tokens'][i]
                    if bool(re.search(r'[a-zA-Z]', text)):  # check English
                        text += ' '
                    tokens.append(examples['tokens'][i])
                    boxes.append(examples['bboxes'][i])
                    original_boxes.append(examples['original_boxes'][i])
                    labels.append(examples['ner_tags'][i][1])
                
                # handel exception
                if len(boxes) == 0 or len(tokens) == 0:
                    continue
                
                # if exp_name == 'FUNSD-r':
                if exp_name != 'CORD-r':
                    # FUNSD format
                    words = []
                    for j in range(len(tokens)):
                        is_chinese = bool(pattern.search(tokens[j]))
                        if is_chinese:
                            for token in tokens[j]:
                                word = {"text": token, "box": original_boxes[j]}
                                words.append(word)
                        else:
                            word = {"text": tokens[j], "box": original_boxes[j]}
                            words.append(word)

                    # transform to entity-level box & labels
                    x1, x2 = min(row[0] for row in original_boxes), max(row[2] for row in original_boxes)
                    y1, y2 = min(row[1] for row in original_boxes), max(row[3] for row in original_boxes)
                    box = [x1, y1, x2, y2]
                    
                    check_consist = all(label == labels[0] for label in labels)  # only check if all the token in the set is the same
                    if ent_id in correct_idx:
                        label = labels[0]
                    else:
                        label = "error"
                        # label = 'other'

                    ocr = {"text": text, "box": box, "label": label, "words": words, "id": id, "linking": []}
                    
                    form.append(ocr)
                elif exp_name == 'CORD-r':
                    # convert label
                    if ent_id in correct_idx:
                        label = labels[0]
                    else:
                        label = "error"
                        # label = 'other'
                    
                    if label in label_convert.keys():
                        label = label_convert[label]
                    
                    # words
                    words = []
                    for i in range(len(tokens)):
                        box = original_boxes[i]
                        quad = {"x2": box[2], "y3": box[3], "x3": box[2], "y4": box[3], "x1": box[0], "y1": box[1], "x4": box[0], "y2": box[1]}
                        word = {"quad": quad, "is_key": 0, "row_id": self.row_id, "text": tokens[i]}
                        words.append(word)
                    
                    entity = {"words": words, "category": label, "group_id": id}
                    valid_line.append(entity)
                    self.row_id += 1

        # save_dir = None
        # write inference file
        prediction = {}
        if save_dir != None:
            if exp_name == 'CORD-r':  # write CORD format
                filename = 'receipt_0' + examples['image_path'][81:-4] + '.json'  # 77 is path dependent
                prediction = {'dontcare': [], 'valid_line': valid_line, 'meta': meta, 'roi': roi, 'repeating_symbol': []}

                save_dir = f"outputs/results/{exp_name}-decode/testing_data"

            else:   # write FUNSD format
                if exp_name == 'leave-one-out':  # special save dir path
                    save_dir = 'outputs/results/CORD-r/testing_data'
                else:
                    save_dir = f"outputs/results/{exp_name}-decode/testing_data"

                filename = os.path.basename(examples['image_path'])[:-3] + 'json'
                print(filename)
                prediction = {"form": form}
            
        
            # write file
            if not os.path.exists(save_dir):
                self.write = True
                os.makedirs(save_dir)
            elif self.write == False:
                shutil.rmtree(save_dir)  # If it exists, remove it along with its contents
                self.write = True
                os.makedirs(save_dir)

            with open(os.path.join(save_dir, filename), 'w', encoding='utf-8') as f:
                json.dump(prediction, f, ensure_ascii=False, indent=4)
        
        return prediction
