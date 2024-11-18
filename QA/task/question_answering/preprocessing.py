import os
import sys
import pickle
import argparse
import pandas as pd
from tqdm.auto import tqdm
import torch
from transformers import AutoTokenizer, AutoConfig
from datasets import load_dataset
# Custom Modules
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.utils import check_path, get_huggingface_model_name

def preprocess_function(examples, tokenizer, max_seq_len):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_seq_len,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

def load_data(args: argparse.Namespace) -> tuple:
    """
    Load data from huggingface datasets.
    If dataset is not in huggingface datasets, takes data from local directory.

    Args:
        args (argparse.Namespace): Arguments.

    Returns:
        train_data (dict): Training data. (context, question, answer_start, answer_end)
        valid_data (dict): Validation data. (context, question, answer_start, answer_end)
        test_data (dict): Test data. (context, question, answer_start, answer_end)
        num_classes (int): Number of classes (this will be None for SQuAD).
    """

    name = args.task_dataset.lower()

    if name == 'squad':
        dataset = load_dataset('squad')

        train_df = pd.DataFrame(dataset['train'])
        valid_df = pd.DataFrame(dataset['validation'])

        valid_size = len(valid_df)

        train_df_subset = train_df.iloc[:-valid_size]
        test_df_subset = train_df.iloc[-valid_size:]

        train_data = train_df_subset.to_dict(orient="list")
        valid_data = valid_df.to_dict(orient="list")
        test_data = test_df_subset.to_dict(orient="list")

    return train_data, valid_data, test_data

def preprocessing(args: argparse.Namespace) -> None:
    """
    Main function for preprocessing.

    Args:
        args (argparse.Namespace): Arguments.
    """

    train_data, valid_data, test_data = load_data(args)

    model_name = get_huggingface_model_name(args.model_type)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)

    data_dict = {
        'train': {
            'input_ids': [],
            'attention_mask': [],
            'token_type_ids': [],
            'start_positions': [],
            'end_positions': [],
            'vocab_size': config.vocab_size,
            'pad_token_id': tokenizer.pad_token_id
        },
        'valid': {
            'input_ids': [],
            'attention_mask': [],
            'token_type_ids': [],
            'start_positions': [],
            'end_positions': [],
            'vocab_size': config.vocab_size,
            'pad_token_id': tokenizer.pad_token_id
        },
        'test': {
            'input_ids': [],
            'attention_mask': [],
            'token_type_ids': [],
            'start_positions': [],
            'end_positions': [],
            'vocab_size': config.vocab_size,
            'pad_token_id': tokenizer.pad_token_id
        }
    }

    preprocessed_path = os.path.join(args.preprocess_path, args.task, args.task_dataset, args.model_type)
    check_path(preprocessed_path)

    for split_data, split in zip([train_data, valid_data, test_data], ['train', 'valid', 'test']):
        processed_data = preprocess_function(split_data, tokenizer, args.max_seq_len)
        
        if 'roberta' not in args.model_type:
            for key in ['input_ids', 'attention_mask', 'token_type_ids', 'start_positions', 'end_positions']:
                data_dict[split][key] = torch.tensor(processed_data[key], dtype=torch.long)
        else:
            for key in ['input_ids', 'attention_mask', 'start_positions', 'end_positions']:
                data_dict[split][key] = torch.tensor(processed_data[key], dtype=torch.long)
            data_dict[split]['token_type_ids'] = torch.zeros(len(processed_data['input_ids']), args.max_seq_len, dtype=torch.long)

        with open(os.path.join(preprocessed_path, f'{split}_processed.pkl'), 'wb') as f:
            pickle.dump(data_dict[split], f)

