# Standard Library Modules
import os
import sys
import json
import pickle
import random
import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) 
import pandas as pd
from transformers import AutoTokenizer, AutoConfig
from tqdm.auto import tqdm
from datasets import load_dataset 
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.utils import check_path, get_huggingface_model_name

def preprocessing(args: argparse.Namespace) -> pd.DataFrame:
    if args.task_dataset == 'cnn_dailymail':
        dataset = load_dataset("abisee/cnn_dailymail", "2.0.0")
    train_df = dataset['train'].to_pandas()
    validation_df = dataset['validation'].to_pandas()
    test_df = dataset['test'].to_pandas()
    data_dict = {
            'train': {
                'source_text': [],
                'target_text': [],
                'source_attention_mask': [],
                'target_attention_mask': []
            },
            'valid': {
                'source_text': [],
                'target_text': [],
                'source_attention_mask': [],
                'target_attention_mask': []
            },
            'test': {
                'source_text': [],
                'target_text': [],
                'source_attention_mask': [],
                'target_attention_mask': []
            }
        }

    preprocessed_path = os.path.join(args.preprocess_path, args.task, args.task_dataset, args.model_type)
    check_path(preprocessed_path)

    model = get_huggingface_model_name(args.model_type)
    tokenizer = AutoTokenizer.from_pretrained(model)
    config = AutoConfig.from_pretrained(model)

    for split_name, split_data in zip(['train', 'valid', 'test'], [train_df, validation_df, test_df]):
        for idx in tqdm(range(len(split_data)), desc=f'Preprocessing {split_name} data'):
            target_text = split_data.loc[idx, 'highlights']
            source_text = "Summarize:" + split_data.loc[idx, 'article']
            
            src_tokenized = tokenizer(source_text, padding='max_length', truncation=True,
                                      max_length=args.max_seq_len, return_tensors='pt')
            with tokenizer.as_target_tokenizer():
                tgt_tokenized = tokenizer(target_text, padding='max_length', truncation=True,
                                          max_length=args.max_seq_len, return_tensors='pt')

            data_dict[split_name]['source_text'].append(src_tokenized['input_ids'].squeeze())
            data_dict[split_name]['target_text'].append(tgt_tokenized['input_ids'].squeeze())
            data_dict[split_name]['source_attention_mask'].append(src_tokenized['attention_mask'].squeeze())
            data_dict[split_name]['target_attention_mask'].append(tgt_tokenized['attention_mask'].squeeze())

    for split in ['train', 'valid', 'test']:
        with open(os.path.join(preprocessed_path, f'{split}.pkl'), 'wb') as f:
            pickle.dump(data_dict[split], f)