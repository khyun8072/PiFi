import os
import sys
import pickle
import argparse
import bs4
import pandas as pd
from tqdm.auto import tqdm
import torch
from transformers import AutoTokenizer, AutoConfig
from datasets import load_dataset
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.utils import check_path, get_huggingface_model_name

def load_data(args: argparse.Namespace) -> tuple: 
    """
    Load data from huggingface datasets.
    If dataset is not in huggingface datasets, takes data from local directory.

    Args:
        dataset_name (str): Dataset name.
        args (argparse.Namespace): Arguments.
        train_valid_split (float): Train-valid split ratio.

    Returns:
        train_data (dict): Training data. (text, label)
        valid_data (dict): Validation data. (text, label)
        test_data (dict): Test data. (text, label)
        num_classes (int): Number of classes.
    """

    name = args.task_dataset.lower()
    train_valid_split = args.train_valid_split

    train_data = {
        'text1': [],
        'text2': [],
        'label': []
    }
    valid_data = {
        'text1': [],
        'text2': [],
        'label': []
    }
    test_data = {
        'text1': [],
        'text2': [],
        'label': []
    }

    if name == 'snli':
        dataset = load_dataset('snli')

        train_df = pd.DataFrame(dataset['train'])
        valid_df = pd.DataFrame(dataset['validation'])
        test_df = pd.DataFrame(dataset['test'])
        num_classes = 3

        train_data['text1'] = train_df['premise'].tolist()
        train_data['text2'] = train_df['hypothesis'].tolist()
        train_data['label'] = train_df['label'].tolist()

        valid_data['text1'] = valid_df['premise'].tolist()
        valid_data['text2'] = valid_df['hypothesis'].tolist()
        valid_data['label'] = valid_df['label'].tolist()

        test_data['text1'] = test_df['premise'].tolist()
        test_data['text2'] = test_df['hypothesis'].tolist()
        test_data['label'] = test_df['label'].tolist()

    elif name == 'mnli':
        dataset = load_dataset('glue', 'mnli')

        train_df = pd.DataFrame(dataset['train'])
        valid_df = pd.DataFrame(dataset['validation_mismatched'])
        test_df = valid_df.copy()

        valid_size = len(valid_df)
        train_valid_split = train_df.iloc[-valid_size:] 
        train_df = train_df.iloc[:-valid_size]  


        num_classes = 3

        train_data['idx'] = train_df['idx'].tolist()
        train_data['text1'] = train_df['premise'].tolist()
        train_data['text2'] = train_df['hypothesis'].tolist()
        train_data['label'] = train_df['label'].tolist()

        valid_data['idx'] = train_valid_split['idx'].tolist()
        valid_data['text1'] = train_valid_split['premise'].tolist()
        valid_data['text2'] = train_valid_split['hypothesis'].tolist()
        valid_data['label'] = train_valid_split['label'].tolist()

        test_data['idx'] = test_df['idx'].tolist()
        test_data['text1'] = test_df['premise'].tolist()
        test_data['text2'] = test_df['hypothesis'].tolist()
        test_data['label'] = test_df['label'].tolist()

    return train_data, valid_data, test_data, num_classes

def preprocessing(args: argparse.Namespace) -> None:
    """
    Main function for preprocessing.

    Args:
        args (argparse.Namespace): Arguments.
    """

    train_data, valid_data, test_data, num_classes = load_data(args)

    model_name = get_huggingface_model_name(args.model_type)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)

    data_dict = {
        'train': {
            'input_ids': [],
            'attention_mask': [],
            'token_type_ids': [],
            'labels': [],
            'num_classes': num_classes,
            'vocab_size': config.vocab_size,
            'pad_token_id': tokenizer.pad_token_id
        },
        'valid': {
            'input_ids': [],
            'attention_mask': [],
            'token_type_ids': [],
            'labels': [],
            'num_classes': num_classes,
            'vocab_size': config.vocab_size,
            'pad_token_id': tokenizer.pad_token_id
        },
        'test': {
            'input_ids': [],
            'attention_mask': [],
            'token_type_ids': [],
            'labels': [],
            'num_classes': num_classes,
            'vocab_size': config.vocab_size,
            'pad_token_id': tokenizer.pad_token_id
        }
    }

    preprocessed_path = os.path.join(args.preprocess_path, args.task, args.task_dataset, args.model_type)
    check_path(preprocessed_path)

    for split_data, split in zip([train_data, valid_data, test_data], ['train', 'valid', 'test']):
        for idx in tqdm(range(len(split_data['text1'])), desc=f'Preprocessing {split} data', position=0, leave=True):
            text1 = split_data['text1'][idx]
            text2 = split_data['text2'][idx]
            label = split_data['label'][idx]
            if args.task_dataset.lower() == 'snli':
                if label not in [0, 1, 2]:
                    continue  
            tokenized = tokenizer(text1, text2, padding='max_length', truncation=True,
                                  max_length=args.max_seq_len, return_tensors='pt')

            data_dict[split]['input_ids'].append(tokenized['input_ids'].squeeze())
            data_dict[split]['attention_mask'].append(tokenized['attention_mask'].squeeze())
            if args.model_type in ['bert', 'albert', 'electra', 'deberta', 'debertav3', 'bert-large']:
                data_dict[split]['token_type_ids'].append(tokenized['token_type_ids'].squeeze())
            else: # roberta does not use token_type_ids
                data_dict[split]['token_type_ids'].append(torch.zeros(args.max_seq_len, dtype=torch.long))
            data_dict[split]['labels'].append(torch.tensor(label, dtype=torch.long))

        # Save data as pickle file
        with open(os.path.join(preprocessed_path, f'{split}_processed.pkl'), 'wb') as f:
            pickle.dump(data_dict[split], f)