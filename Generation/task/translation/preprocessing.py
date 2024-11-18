import os
import sys
import pickle
import argparse
import bs4
import pandas as pd
from tqdm.auto import tqdm
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig

from datasets import load_dataset
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.utils import check_path, get_huggingface_model_name

def load_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
    return lines

def load_data(args: argparse.Namespace):

    name = args.task_dataset.lower()


    dataset = load_dataset("bentrevett/multi30k", split=['train', 'validation', 'test'])
    train_data = dataset[0]
    valid_data = dataset[1]
    test_data = dataset[2]

    return train_data, valid_data, test_data

def preprocessing(args: argparse.Namespace) -> None:

    train_data, valid_data, test_data = load_data(args)

    model = get_huggingface_model_name(args.model_type)
    tokenizer = AutoTokenizer.from_pretrained(model)
    config = AutoConfig.from_pretrained(model)


    data_dict = {
        'train': {
            'EN_text_ids': [],
            'DE_text_ids': [],
            'src_attention_mask': [],
            'tgt_attention_mask': [],
            'src_vocab_size': config.vocab_size,
            'tgt_vocab_size': config.vocab_size
        },
        'valid': {
            'EN_text_ids': [],
            'DE_text_ids': [],
            'src_attention_mask': [],
            'tgt_attention_mask': [],
            'src_vocab_size': config.vocab_size,
            'tgt_vocab_size': config.vocab_size
        },
        'test': {
            'EN_text_ids': [],
            'DE_text_ids': [],
            'src_attention_mask': [],
            'tgt_attention_mask': [],
            'src_vocab_size': config.vocab_size,
            'tgt_vocab_size': config.vocab_size
        }
    }

    preprocessed_path = os.path.join(args.preprocess_path, args.task, args.task_dataset, args.model_type)
    check_path(preprocessed_path)

    prefix = "translate English to French: "    
    for split_data, split in zip([train_data, valid_data, test_data], ['train', 'valid', 'test']):
        for idx in tqdm(range(len(split_data['en'])), desc= 'Preprocessing', position=0, leave=True):
            
            src = prefix + split_data['en'][idx]
            tgt = split_data['de'][idx]

            src_tokenized = tokenizer(src, padding='max_length', truncation=True,
                               max_length=args.max_seq_len, return_tensors='pt')
            with tokenizer.as_target_tokenizer():
                tgt_tokenized = tokenizer(tgt, padding='max_length', truncation=True,
                                max_length=args.max_seq_len, return_tensors='pt')
                
            data_dict[split]['EN_text_ids'].append(src_tokenized['input_ids'].squeeze())
            data_dict[split]['DE_text_ids'].append(tgt_tokenized['input_ids'].squeeze())
            data_dict[split]['src_attention_mask'].append(src_tokenized['attention_mask'].squeeze())
            data_dict[split]['tgt_attention_mask'].append(tgt_tokenized['attention_mask'].squeeze())

        with open(os.path.join(preprocessed_path, f'{split}_processed.pkl'), 'wb') as f:
            pickle.dump(data_dict[split], f)