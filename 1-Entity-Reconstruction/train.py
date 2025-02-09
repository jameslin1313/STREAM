"""
Date: 2021-05-31 19:50:58
LastEditors: GodK
"""

import os
import config
import sys
import torch
import json
from transformers import BertTokenizerFast, BertModel
from common.utils import Preprocessor, multilabel_categorical_crossentropy
from models.GlobalPointer import DataMaker, MyDataset, GlobalPointer, MetricsCalculator
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import glob

# import wandb
from evaluate import load_model
import time

from dataclasses import dataclass, field
from typing import Optional

from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    AutoModel
)
from common.utils import DocumentLoader


# LayoutLMv3 Config
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="microsoft/layoutlmv3-base", metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: Optional[str] = field(default="ner", metadata={"help": "The name of the task (ner, pos...)."})
    dataset_name: Optional[str] = field(
        default='funsd', metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a csv or JSON file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to predict on (a csv or JSON file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
            "value if set."
        },
    )
    label_all_tokens: bool = field(
        default=False,
        metadata={
            "help": "Whether to put the label for one word on all tokens of generated by that word or just on the "
            "one (in which case the other tokens will have a padding index)."
        },
    )
    return_entity_level_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to return all the entity levels during evaluation or just the overall ones."},
    )
    segment_level_layout: bool = field(default=True)
    visual_embed: bool = field(default=True)
    data_dir: Optional[str] = field(default=None)
    input_size: int = field(default=224, metadata={"help": "images input size for backbone"})
    second_input_size: int = field(default=112, metadata={"help": "images input size for discrete vae"})
    train_interpolation: str = field(
        default='bicubic', metadata={"help": "Training interpolation (random, bilinear, bicubic)"})
    second_interpolation: str = field(
        default='lanczos', metadata={"help": "Interpolation for discrete vae (random, bilinear, bicubic)"})
    imagenet_default_mean_and_std: bool = field(default=False, metadata={"help": ""})


config = config.train_config
hyper_parameters = config["hyper_parameters"]

os.environ["TOKENIZERS_PARALLELISM"] = "true"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
config["num_workers"] = 0 if sys.platform.startswith("linux") else 0

# for reproductivity
torch.manual_seed(hyper_parameters["seed"])  # pytorch random seed
torch.backends.cudnn.deterministic = True

model_state_dict_dir = os.path.join(config["path_to_save_model"], config["exp_name"],
                                    time.strftime("%Y-%m-%d_%H.%M.%S", time.gmtime()))
if not os.path.exists(model_state_dict_dir):
    os.makedirs(model_state_dict_dir)

entity_grouped_label = config['entity_grouped_label']


def load_data(data_path, data_type="train", exp_name='cluener'):
    """读取数据集

    Args:
        data_path (str): 数据存放路径
        data_type (str, optional): 数据类型. Defaults to "train".

    Returns:
        (json): train和valid中一条数据格式：{"text":"","entity_list":[(start, end, label), (start, end, label)...]}
    """
    if data_type == "train" or data_type == "valid":
        datas = []
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                line = json.loads(line)
                item = {}
                item["text"] = line["text"]
                item["entity_list"] = []
                for k, v in line['label'].items():
                    for spans in v.values():
                        if exp_name == 'cluener':  # only record start, end
                            for start, end in spans:
                                item["entity_list"].append((start, end, k))
                datas.append(item)
        return datas
    else:
        return json.load(open(data_path, encoding="utf-8"))


# load NER types
ent2id_path = os.path.join(config["data_home"], config["exp_name"], config["ent2id"])
ent2id = load_data(ent2id_path, "ent2id")
ent_type_size = len(ent2id)

max_seq_len = hyper_parameters["max_seq_len"]


# LayoutLMv3 arguments
parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    # If we pass only one argument to the script and it's the path to a json file,
    # let's parse it to get our arguments.
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
else:
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
print('Backbones: ', model_args.model_name_or_path)


# set up tokenizer

# BertTokenizerFast                  : dict_keys(['input_ids', 'token_type_ids', 'attention_mask'])
bert_tokenizer = BertTokenizerFast.from_pretrained(config["bert_path"], add_special_tokens=True, do_lower_case=False)

# "microsoft/layoutlmv3-base-chinese": dict_keys(['input_ids', 'attention_mask'])
tokenizer = AutoTokenizer.from_pretrained(
        # "microsoft/layoutlmv3-base-chinese",
        model_args.model_name_or_path,  # input should be a list of tokens
        tokenizer_file=None,  # avoid loading from a cached file of the pre-trained model in another machine
        use_fast=True,
        add_prefix_space=True
    )


def load_document_data(documentloader, data_type='train', exp_name='FUNSD-r', visual_embed=True):
    documents = documentloader.load_document_list(data_type)

    # original GP
    ent_type_size = len(ent2id)  # 实体类别

    all_inputs = []
    for i, sample in enumerate(documents):
        # {'input_ids', 'attention_msk', 'overflow_to_sample_mapping', 'bbox', 'labels', 'images'}
        if exp_name == 'FUNSD-r' or exp_name == 'CORD-r' or exp_name == 'EC-FUNSD-r':  # English
            # tokenized_inputs = documentloader.tokenize_and_align_labels(sample[1], ent2id, visual_embed=visual_embed, max_seq_len=max_seq_len, data_type=data_type, img_id=i, exp_name=exp_name)
            tokenized_inputs = documentloader.tokenize_and_align_labels(sample[1], ent2id, visual_embed=visual_embed, max_seq_len=max_seq_len, data_type=data_type, img_id=i, exp_name=exp_name, entity_grouped_label=entity_grouped_label)
        else:  # Chinese
            tokenized_inputs = documentloader.tokenize_and_align_labels_chinese(sample[1], ent2id, visual_embed=visual_embed, max_seq_len=max_seq_len, data_type=data_type, img_id=i, exp_name=exp_name, entity_grouped_label=entity_grouped_label)  # {'input_ids', 'attention_msk', 'overflow_to_sample_mapping', 'bbox', 'labels', 'images'}
        
        input_ids = torch.tensor(tokenized_inputs["input_ids"]).long()
        attention_mask = torch.tensor(tokenized_inputs["attention_mask"]).long()
        bbox = torch.tensor(tokenized_inputs["bbox"])  # (N, 4) with top-left, right-bottom
        
        if tokenized_inputs['labels'] is not None:
            labels = torch.tensor(tokenized_inputs["labels"]).long()
            paths = tokenized_inputs["paths"]  # entity grouping labels
            entity_ser = tokenized_inputs["entity_ser"]

        images = None
        if visual_embed:
            images = torch.unsqueeze(tokenized_inputs['images'][0], dim=0)

        sample_input = (sample, input_ids, attention_mask, bbox, images, labels, paths, entity_ser)

        all_inputs.append(sample_input)

    return all_inputs


def data_generator(data_type="train", exp_name='FUNSD-r'):
    """
    读取数据，生成DataLoader。
    """
    # set visual embed
    visual_embed = True
    
    # data_type = 'valid'  # for eval
    # data_type = 'temp'   # for train

    documentloader = DocumentLoader(tokenizer, exp_name=exp_name)
    if data_type == 'train':
        train_data = load_document_data(documentloader, data_type='train', visual_embed=visual_embed, exp_name=exp_name)
        valid_data = load_document_data(documentloader, data_type='test', visual_embed=visual_embed, exp_name=exp_name)
    if data_type == 'test':
        valid_data = load_document_data(documentloader, data_type='test', visual_embed=visual_embed, exp_name=exp_name)
    elif data_type == 'valid':
        valid_data = load_document_data(documentloader, data_type='temp-2', visual_embed=visual_embed, exp_name=exp_name)
    elif data_type == 'temp':
        train_data = load_document_data(documentloader, data_type='long', visual_embed=visual_embed, exp_name=exp_name)
        valid_data = load_document_data(documentloader, data_type='long', visual_embed=visual_embed, exp_name=exp_name)

    data_maker = DataMaker(tokenizer)

    if data_type == "train" or data_type == 'temp':
        train_dataloader = DataLoader(MyDataset(train_data),
                                      batch_size=hyper_parameters["batch_size"],
                                      shuffle=True,
                                      num_workers=config["num_workers"],
                                      drop_last=False,
                                      collate_fn=lambda x: data_maker.generate_batch(x, max_seq_len, ent2id)
                                      )
        valid_dataloader = DataLoader(MyDataset(valid_data),
                                      batch_size=hyper_parameters["batch_size"],
                                      shuffle=True,
                                      num_workers=config["num_workers"],
                                      drop_last=False,
                                      collate_fn=lambda x: data_maker.generate_batch(x, max_seq_len, ent2id)
                                      )
        return train_dataloader, valid_dataloader
    else:
        valid_dataloader = DataLoader(MyDataset(valid_data),
                                      batch_size=hyper_parameters["batch_size"],
                                      shuffle=True,
                                      num_workers=config["num_workers"],
                                      drop_last=False,
                                      collate_fn=lambda x: data_maker.generate_batch(x, max_seq_len, ent2id)
                                      )
        return valid_dataloader


metrics = MetricsCalculator()


def train_step(batch_train, model, optimizer, criterion):
    batch_samples, batch_input_ids, batch_attention_mask, batch_bbox, batch_image, batch_labels, batch_paths, batch_entity_ser = batch_train

    if batch_image == None:  # not use image
        batch_input_ids, batch_attention_mask, batch_bbox, batch_labels = (batch_input_ids.to(device),
                                                                           batch_attention_mask.to(device),
                                                                           batch_bbox.to(device),
                                                                           batch_labels.to(device)
                                                                          )
    else:
        batch_input_ids, batch_attention_mask, batch_bbox, batch_image, batch_labels = (batch_input_ids.to(device),
                                                                                        batch_attention_mask.to(device),
                                                                                        batch_bbox.to(device),
                                                                                        batch_image.to(device),
                                                                                        batch_labels.to(device)
                                                                                       )
    
    batch_size = batch_input_ids.shape[0]
    
    loss = 0

    for i in range(batch_size):
        logits = model(batch_input_ids[i].unsqueeze(0), batch_bbox[i].unsqueeze(0), batch_attention_mask[i].unsqueeze(0), batch_image)
        loss = criterion(batch_labels[i], logits)

        # print(stop)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()


# set up encoder

# LayoutLMv3
layoutlmconfig = AutoConfig.from_pretrained(
    model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    # "microsoft/layoutlmv3-base-chinese",
    num_labels=4,
    finetuning_task=data_args.task_name,
    # cache_dir=model_args.cache_dir,
    revision=model_args.model_revision,
    input_size=data_args.input_size,
    use_auth_token=True if model_args.use_auth_token else None,
)

encoder = AutoModel.from_pretrained(
    model_args.model_name_or_path,
    # "microsoft/layoutlmv3-base-chinese",
    from_tf=bool(".ckpt" in model_args.model_name_or_path),
    config=layoutlmconfig,
    # cache_dir=model_args.cache_dir,
    revision=model_args.model_revision,
    use_auth_token=True if model_args.use_auth_token else None,
)

model = GlobalPointer(encoder, ent_type_size, config['inner_dim'])
model = model.to(device)


def train(model, dataloader, epoch, optimizer):
    model.train()

    # loss func
    def loss_fun(y_true, y_pred):
        """
        y_true:(batch_size, ent_type_size, seq_len, seq_len)
        y_pred:(batch_size, ent_type_size, seq_len, seq_len)
        """
        batch_size, ent_type_size = y_pred.shape[:2]
        y_true = y_true.reshape(batch_size * ent_type_size, -1)
        y_pred = y_pred.reshape(batch_size * ent_type_size, -1)

        if entity_grouped_label:
            loss = multilabel_categorical_crossentropy(y_true[0], y_pred[0]) * 0.8 + multilabel_categorical_crossentropy(y_true[1], y_pred[1]) * 0.2
        else:
            loss = multilabel_categorical_crossentropy(y_true, y_pred)
        
        return loss

    # scheduler
    if hyper_parameters["scheduler"] == "CAWR":
        T_mult = hyper_parameters["T_mult"]
        rewarm_epoch_num = hyper_parameters["rewarm_epoch_num"]
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                         len(train_dataloader) * rewarm_epoch_num,
                                                                         T_mult)
    elif hyper_parameters["scheduler"] == "Step":
        decay_rate = hyper_parameters["decay_rate"]
        decay_steps = hyper_parameters["decay_steps"]
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_steps, gamma=decay_rate)
    else:
        scheduler = None

    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    total_loss = 0.
    for batch_ind, batch_data in pbar:

        loss = train_step(batch_data, model, optimizer, loss_fun)

        total_loss += loss

        avg_loss = total_loss / (batch_ind + 1)
        if scheduler is not None:
            scheduler.step()

        pbar.set_description(
            f'Project:{config["exp_name"]}, Epoch: {epoch + 1}/{hyper_parameters["epochs"]}, Step: {batch_ind + 1}/{len(dataloader)}')
        pbar.set_postfix(loss=avg_loss, lr=optimizer.param_groups[0]["lr"])


def valid_step(batch_valid, model, run_type='train', exp_name='FUNSD-r', save_dir=None, count=0):
    batch_samples, batch_input_ids, batch_attention_mask, batch_bbox, batch_image, batch_labels, batch_paths, batch_entity_ser = batch_valid

    if batch_image == None:
        batch_input_ids, batch_attention_mask, batch_bbox, batch_labels = (batch_input_ids.to(device),
                                                                           batch_attention_mask.to(device),
                                                                           batch_bbox.to(device),
                                                                           batch_labels.to(device)
                                                                          )
    else:
        batch_input_ids, batch_attention_mask, batch_bbox, batch_image, batch_labels = (batch_input_ids.to(device),
                                                                                        batch_attention_mask.to(device),
                                                                                        batch_bbox.to(device),
                                                                                        batch_image.to(device),
                                                                                        batch_labels.to(device)
                                                                                        )

    batch_size = batch_input_ids.shape[0]

    full_logits = []
    sample_f1, sample_precision, sample_recall = 0, 0, 0
    entity_precision, entity_recall, entity_f1, correct_ids = 0, 0, 0, []
    decodes = []

    with torch.no_grad():
        for i in range(batch_size):
            logits = model(batch_input_ids[i].unsqueeze(0), batch_bbox[i].unsqueeze(0), batch_attention_mask[i].unsqueeze(0), batch_image)        
            full_logits.append(logits)

            # token-level metrics
            f1, precision, recall = metrics.get_evaluate_fpr(logits, batch_labels[i])
            sample_f1 += f1 / batch_size
            sample_precision += precision / batch_size
            sample_recall += recall / batch_size

            # decode to ent path
            pred_decode = model.decode(logits[:,0,:,:].unsqueeze(0))
            decodes.append(pred_decode)

            # entity-level metrics
            ent_precision, ent_recall, ent_f1, correct_idx = metrics.get_entity_eval(pred_decode, batch_paths[i], batch_entity_ser, count=count)
            entity_f1 += ent_f1 / batch_size
            entity_precision += ent_precision / batch_size
            entity_recall += ent_recall / batch_size
            correct_ids.append(correct_idx)
    
        if run_type == 'eval':  # inference
            # prediction = model.inference(batch_paths, tokenizer, batch_samples, exp_name=exp_name, save_dir=save_dir, correct_ids=correct_ids)
            prediction = model.inference(decodes, tokenizer, batch_samples, exp_name=exp_name, save_dir=save_dir, correct_ids=correct_ids)

        return sample_f1 , sample_precision, sample_recall, entity_precision, entity_recall, entity_f1
    
    return sample_f1, sample_precision, sample_recall


def valid(model, dataloader, run_type='train', exp_name='FUNSD-r', save_dir=None):
    model.eval()
    
    total_f1, total_precision, total_recall = 0., 0., 0.
    total_ent_f1, total_ent_precision, total_ent_recall = 0., 0., 0.
    count = 0
    for batch_data in tqdm(dataloader, desc="Validating"):
        if run_type == 'train' or run_type == 'eval':
            f1, precision, recall, ent_precision, ent_recall, ent_f1 = valid_step(batch_data, model, run_type, exp_name=exp_name, save_dir=save_dir, count=count)
            count += 1
        
            total_ent_f1 += ent_f1
            total_ent_precision += ent_precision
            total_ent_recall += ent_recall

        total_f1 += f1
        total_precision += precision
        total_recall += recall

        
    avg_f1 = total_f1 / (len(dataloader))
    avg_precision = total_precision / (len(dataloader))
    avg_recall = total_recall / (len(dataloader))
    print("******************************************")
    print(f'avg_precision: {avg_precision}, avg_recall: {avg_recall}, avg_f1: {avg_f1}')
    print("******************************************")
    
    if run_type == 'eval' or run_type == 'train':
        avg_ent_f1 = total_ent_f1 / (len(dataloader))
        avg_ent_precision = total_ent_precision / (len(dataloader))
        avg_ent_recall = total_ent_recall / (len(dataloader))
        print("==========================================")
        print(f'avg_precision: {avg_ent_precision}, avg_recall: {avg_ent_recall}, avg_f1: {avg_ent_f1}')
        print("==========================================")
    
    return avg_ent_f1
    


if __name__ == '__main__':
    if config["run_type"] == "train":
        # model = load_model()

        train_dataloader, valid_dataloader = data_generator(exp_name=config['exp_name'])

        # optimizer
        init_learning_rate = float(hyper_parameters["lr"])
        optimizer = torch.optim.Adam(model.parameters(), lr=init_learning_rate)
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('trainable parameter: ', pytorch_total_params)

        max_f1 = 0.
        for epoch in range(hyper_parameters["epochs"]):
            train(model, train_dataloader, epoch, optimizer)
            valid_f1 = valid(model, valid_dataloader, run_type=config['run_type'], exp_name=config['exp_name'])
            if valid_f1 > max_f1:
                max_f1 = valid_f1
                if valid_f1 > config["f1_2_save"]:  # save the best model
                    model_state_num = len(glob.glob(model_state_dict_dir + "/model_state_dict_*.pt"))
                    torch.save(model.state_dict(),
                               os.path.join(model_state_dict_dir, "model_state_dict_{}.pt".format(model_state_num)))
            print(f"Best F1: {max_f1}")
            print("******************************************")

    elif config["run_type"] == "eval":
        # 此处的 eval 是为了评估测试集的 p r f1（如果测试集有标签的情况），无标签预测使用 evaluate.py
        model = load_model()

        test_dataloader = data_generator(data_type="test", exp_name=config['exp_name'])
        documentloader = documentloader = DocumentLoader(tokenizer, exp_name=config['exp_name'])

        save_dir = os.path.join(config['save_dir'], config['exp_name'])

        valid(model, test_dataloader, run_type=config['run_type'], exp_name=config['exp_name'], save_dir=save_dir)
