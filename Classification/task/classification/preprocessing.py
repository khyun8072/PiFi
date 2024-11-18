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
from sklearn.model_selection import train_test_split

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
        'text': [],
        'label': []
    }
    valid_data = {
        'text': [],
        'label': []
    }
    test_data = {
        'text': [],
        'label': []
    }

    if name == 'sst2':
        dataset = load_dataset('SetFit/sst2')

        train_df = pd.DataFrame(dataset['train'])
        valid_df = pd.DataFrame(dataset['validation'])
        test_df = pd.DataFrame(dataset['test'])
        num_classes = 2

        train_data['text'] = train_df['text'].tolist()
        train_data['label'] = train_df['label'].tolist()
        valid_data['text'] = valid_df['text'].tolist()
        valid_data['label'] = valid_df['label'].tolist()
        test_data['text'] = test_df['text'].tolist()
        test_data['label'] = test_df['label'].tolist()
    if name == 'sst5':
        dataset = load_dataset('SetFit/sst5')

        train_df = pd.DataFrame(dataset['train'])
        valid_df = pd.DataFrame(dataset['validation'])
        test_df = pd.DataFrame(dataset['test'])
        num_classes = 5

        train_data['text'] = train_df['text'].tolist()
        train_data['label'] = train_df['label'].tolist()
        valid_data['text'] = valid_df['text'].tolist()
        valid_data['label'] = valid_df['label'].tolist()
        test_data['text'] = test_df['text'].tolist()
        test_data['label'] = test_df['label'].tolist()
    elif name == 'cola':
        dataset = load_dataset("nyu-mll/glue", "cola")

        train_df = pd.DataFrame(dataset['train'])
        test_df = pd.DataFrame(dataset['validation'])
        num_classes = 2

        train_df = train_df.sample(frac=1).reset_index(drop=True) 
        valid_df = train_df[:int(len(train_df) * train_valid_split)]
        train_df = train_df[int(len(train_df) * train_valid_split):]

        train_data['text'] = train_df['sentence'].tolist()
        train_data['label'] = train_df['label'].tolist()
        valid_data['text'] = valid_df['sentence'].tolist()
        valid_data['label'] = valid_df['label'].tolist()
        test_data['text'] = test_df['sentence'].tolist()
        test_data['label'] = test_df['label'].tolist()
    elif name == 'imdb':
        dataset = load_dataset('imdb')

        train_df = pd.DataFrame(dataset['train'])
        test_df = pd.DataFrame(dataset['test'])
        num_classes = 2

        train_df = train_df.sample(frac=1).reset_index(drop=True)
        valid_df = train_df[:int(len(train_df) * train_valid_split)]
        train_df = train_df[int(len(train_df) * train_valid_split):]

        train_data['text'] = train_df['text'].tolist()
        train_data['label'] = train_df['label'].tolist()
        valid_data['text'] = valid_df['text'].tolist()
        valid_data['label'] = valid_df['label'].tolist()
        test_data['text'] = test_df['text'].tolist()
        test_data['label'] = test_df['label'].tolist()
    elif name == 'trec':
        dataset = load_dataset('trec')

        train_df = pd.DataFrame(dataset['train'])
        test_df = pd.DataFrame(dataset['test'])
        num_classes = 6

        train_df = train_df.sample(frac=1).reset_index(drop=True)
        valid_df = train_df[:int(len(train_df) * train_valid_split)]
        train_df = train_df[int(len(train_df) * train_valid_split):]

        train_data['text'] = train_df['text'].tolist()
        train_data['label'] = train_df['coarse_label'].tolist()
        valid_data['text'] = valid_df['text'].tolist()
        valid_data['label'] = valid_df['coarse_label'].tolist()
        test_data['text'] = test_df['text'].tolist()
        test_data['label'] = test_df['coarse_label'].tolist()
    elif name == 'subj':
        dataset = load_dataset('SetFit/subj')

        train_df = pd.DataFrame(dataset['train'])
        test_df = pd.DataFrame(dataset['test'])
        num_classes = 2

        train_df = train_df.sample(frac=1).reset_index(drop=True)
        valid_df = train_df[:int(len(train_df) * train_valid_split)]
        train_df = train_df[int(len(train_df) * train_valid_split):]

        train_data['text'] = train_df['text'].tolist()
        train_data['label'] = train_df['label'].tolist()
        valid_data['text'] = valid_df['text'].tolist()
        valid_data['label'] = valid_df['label'].tolist()
        test_data['text'] = test_df['text'].tolist()
        test_data['label'] = test_df['label'].tolist()
    elif name == 'agnews':
        dataset = load_dataset('ag_news')

        train_df = pd.DataFrame(dataset['train'])
        test_df = pd.DataFrame(dataset['test'])
        num_classes = 4

        train_df = train_df.sample(frac=1).reset_index(drop=True)
        valid_df = train_df[:int(len(train_df) * train_valid_split)]
        train_df = train_df[int(len(train_df) * train_valid_split):]

        train_data['text'] = train_df['text'].tolist()
        train_data['label'] = train_df['label'].tolist()
        valid_data['text'] = valid_df['text'].tolist()
        valid_data['label'] = valid_df['label'].tolist()
        test_data['text'] = test_df['text'].tolist()
        test_data['label'] = test_df['label'].tolist()
    elif name == 'mr':
        train_path = os.path.join(args.data_path, 'MR', 'train.csv')
        test_path = os.path.join(args.data_path, 'MR', 'test.csv')
        train_df = pd.read_csv(train_path, header=None)
        test_df = pd.read_csv(test_path, header=None)
        num_classes = 2

        train_df = train_df.sample(frac=1).reset_index(drop=True)
        valid_df = train_df[:int(len(train_df) * train_valid_split)]
        train_df = train_df[int(len(train_df) * train_valid_split):]

        train_data['text'] = train_df[1].tolist()
        train_data['label'] = train_df[0].tolist()
        valid_data['text'] = valid_df[1].tolist()
        valid_data['label'] = valid_df[0].tolist()
        test_data['text'] = test_df[1].tolist()
        test_data['label'] = test_df[0].tolist()
    elif name == 'cr':
        dataset = load_dataset('SetFit/SentEval-CR')

        train_df = pd.DataFrame(dataset['train'])
        test_df = pd.DataFrame(dataset['test'])
        num_classes = 2

        train_df = train_df.sample(frac=1).reset_index(drop=True)
        valid_df = train_df[:int(len(train_df) * train_valid_split)]
        train_df = train_df[int(len(train_df) * train_valid_split):]

        train_data['text'] = train_df['text'].tolist()
        train_data['label'] = train_df['label'].tolist()
        valid_data['text'] = valid_df['text'].tolist()
        valid_data['label'] = valid_df['label'].tolist()
        test_data['text'] = test_df['text'].tolist()
        test_data['label'] = test_df['label'].tolist()
    elif name == 'proscons':
        train_path = os.path.join(args.data_path, 'ProsCons', 'train.csv')
        test_path = os.path.join(args.data_path, 'ProsCons', 'test.csv')
        train_df = pd.read_csv(train_path, header=None)
        test_df = pd.read_csv(test_path, header=None)
        num_classes = 2

        train_df = train_df.sample(frac=1).reset_index(drop=True)
        valid_df = train_df[:int(len(train_df) * train_valid_split)]
        train_df = train_df[int(len(train_df) * train_valid_split):]

        train_data['text'] = train_df[1].tolist()
        train_data['label'] = train_df[0].tolist()
        valid_data['text'] = valid_df[1].tolist()
        valid_data['label'] = valid_df[0].tolist()
        test_data['text'] = test_df[1].tolist()
        test_data['label'] = test_df[0].tolist()
    elif name == 'dbpedia':
        dataset = load_dataset('dbpedia_14')

        train_df = pd.DataFrame(dataset['train'])
        test_df = pd.DataFrame(dataset['test'])
        num_classes = 14

        train_df = train_df.sample(frac=1).reset_index(drop=True)
        valid_df = train_df[:int(len(train_df) * train_valid_split)]
        train_df = train_df[int(len(train_df) * train_valid_split):]

        train_data['text'] = train_df['content'].tolist()
        train_data['label'] = train_df['label'].tolist()
        valid_data['text'] = valid_df['content'].tolist()
        valid_data['label'] = valid_df['label'].tolist()
        test_data['text'] = test_df['content'].tolist()
        test_data['label'] = test_df['label'].tolist()
    elif name == 'tweet_offensive':
        dataset = load_dataset("cardiffnlp/tweet_eval", 'offensive')

        train_df = pd.DataFrame(dataset['train'])
        valid_df = pd.DataFrame(dataset['validation']) 
        test_df = pd.DataFrame(dataset['test'])
        num_classes = 2

        train_data['text'] = train_df['text'].tolist()
        train_data['label'] = train_df['label'].tolist()
        valid_data['text'] = valid_df['text'].tolist()
        valid_data['label'] = valid_df['label'].tolist()
        test_data['text'] = test_df['text'].tolist()
        test_data['label'] = test_df['label'].tolist()
    elif name == 'yelp_polarity':
        dataset = load_dataset('yelp_polarity')

        train_df = pd.DataFrame(dataset['train'])
        test_df = pd.DataFrame(dataset['test'])
        num_classes = 2

        train_df = train_df.sample(frac=1).reset_index(drop=True)
        valid_df = train_df[:int(len(train_df) * train_valid_split)]
        train_df = train_df[int(len(train_df) * train_valid_split):]

        train_data['text'] = train_df['text'].tolist()
        train_data['label'] = train_df['label'].tolist()
        valid_data['text'] = valid_df['text'].tolist()
        valid_data['label'] = valid_df['label'].tolist()
        test_data['text'] = test_df['text'].tolist()
        test_data['label'] = test_df['label'].tolist()
    elif name == 'tweet_sentiment_binary':
        dataset = load_dataset('tweet_eval', name='sentiment')

        train_df = pd.DataFrame(dataset['train'])
        valid_df = pd.DataFrame(dataset['validation'])
        test_df = pd.DataFrame(dataset['test'])
        num_classes = 2

        train_df = train_df[train_df['label'] != 1]
        valid_df = valid_df[valid_df['label'] != 1]
        test_df = test_df[test_df['label'] != 1]

        train_df['label'] = [1 if label == 2 else 0 for label in train_df['label']]
        valid_df['label'] = [1 if label == 2 else 0 for label in valid_df['label']]
        test_df['label'] = [1 if label == 2 else 0 for label in test_df['label']]

        train_data['text'] = train_df['text'].tolist()
        train_data['label'] = train_df['label'].tolist()
        valid_data['text'] = valid_df['text'].tolist()
        valid_data['label'] = valid_df['label'].tolist()
        test_data['text'] = test_df['text'].tolist()
        test_data['label'] = test_df['label'].tolist()
    elif name == 'yelp_full':
        dataset = load_dataset('yelp_review_full')

        train_df = pd.DataFrame(dataset['train'])
        test_df = pd.DataFrame(dataset['test'])
        num_classes = 5

        train_df = train_df.sample(frac=1).reset_index(drop=True)
        valid_df = train_df[:int(len(train_df) * train_valid_split)]
        train_df = train_df[int(len(train_df) * train_valid_split):]

        train_data['text'] = train_df['text'].tolist()
        train_data['label'] = train_df['label'].tolist()
        valid_data['text'] = valid_df['text'].tolist()
        valid_data['label'] = valid_df['label'].tolist()
        test_data['text'] = test_df['text'].tolist()
        test_data['label'] = test_df['label'].tolist()
    elif name == 'yahoo_answers_title':
        dataset = load_dataset('yahoo_answers_topics')

        train_df = pd.DataFrame(dataset['train'])
        test_df = pd.DataFrame(dataset['test'])
        num_classes = 10

        train_df = train_df.sample(frac=1).reset_index(drop=True)
        valid_df = train_df[:int(len(train_df) * train_valid_split)]
        train_df = train_df[int(len(train_df) * train_valid_split):]

        train_data['text'] = train_df['question_title'].tolist()
        train_data['label'] = train_df['topic'].tolist()
        valid_data['text'] = valid_df['question_title'].tolist()
        valid_data['label'] = valid_df['topic'].tolist()
        test_data['text'] = test_df['question_title'].tolist()
        test_data['label'] = test_df['topic'].tolist()
    elif name == 'yahoo_answers_full':
        dataset = load_dataset('yahoo_answers_topics')

        train_df = pd.DataFrame(dataset['train'])
        test_df = pd.DataFrame(dataset['test'])
        num_classes = 10

        train_df = train_df.sample(frac=1).reset_index(drop=True)
        valid_df = train_df[:int(len(train_df) * train_valid_split)]
        train_df = train_df[int(len(train_df) * train_valid_split):]

        for i in range(len(train_df)):
            train_data['text'].append(train_df['question_title'][i] + '[SEP]' + train_df['question_content'][i] + '[SEP]' + train_df['best_answer'][i])
            train_data['label'].append(train_df['topic'][i])
        for i in range(len(valid_df)):
            valid_data['text'].append(valid_df['question_title'][i] + '[SEP]' + valid_df['question_content'][i] + '[SEP]' + valid_df['best_answer'][i])
            valid_data['label'].append(valid_df['topic'][i])
        for i in range(len(test_df)):
            test_data['text'].append(test_df['question_title'][i] + '[SEP]' + test_df['question_content'][i] + '[SEP]' + test_df['best_answer'][i])
            test_data['label'].append(test_df['topic'][i])
    elif name == 'nsmc':
        dataset = load_dataset('e9t/nsmc', trust_remote_code=True)

        train_df = pd.DataFrame(dataset['train'])
        test_df = pd.DataFrame(dataset['test'])
        num_classes = 2

        train_df = train_df.sample(frac=1).reset_index(drop=True)
        valid_df = train_df[:int(len(train_df) * train_valid_split)]
        train_df = train_df[int(len(train_df) * train_valid_split):]

        train_data['text'] = train_df['document'].tolist()
        train_data['label'] = train_df['label'].tolist()
        valid_data['text'] = valid_df['document'].tolist()
        valid_data['label'] = valid_df['label'].tolist()
        test_data['text'] = test_df['document'].tolist()
        test_data['label'] = test_df['label'].tolist()

    elif name == 'filmstarts':
        def map_labels(label):
            if label in [0, 1, 2]:
                return 0  
            elif label in [4, 5]:
                return 1  
            else:
                return None  

        # 데이터 로드 및 전처리
        file_path = f'{args.data_path}german_classification/filmstarts/filmstarts.tsv'
        column_names = ['address', 'label', 'text']
        data = pd.read_csv(file_path, names=column_names, delimiter='\t', on_bad_lines='skip')
        data = data.dropna(subset=['label', 'text'])

        data['label'] = data['label'].astype(int)
        data['label'] = data['label'].apply(map_labels)

        data = data.dropna(subset=['label'])

        data = data.sample(frac=1).reset_index(drop=True)
        num_classes = len(data['label'].unique())

        train_df, valid_test_data = train_test_split(data, test_size=(train_valid_split), random_state=42)
        valid_df, test_df = train_test_split(valid_test_data, test_size=0.5, random_state=42)

        train_data['text'] = train_df['text'].tolist()
        train_data['label'] = train_df['label'].tolist()
        valid_data['text'] = valid_df['text'].tolist()
        valid_data['label'] = valid_df['label'].tolist()
        test_data['text'] = test_df['text'].tolist()
        test_data['label'] = test_df['label'].tolist()
    
    elif name == 'chinese_toxicity':
        dataset = load_dataset('textdetox/multilingual_toxicity_dataset')
        dataset = dataset['zh']
        df = pd.DataFrame(dataset)
        df = df.sample(frac=1).reset_index(drop=True)
        num_classes = len(df['toxic'].unique())
        valid_test_df = df[:int(len(df) * train_valid_split)]
        train_df = df[int(len(df) * train_valid_split):]
        half_valid_idx = len(valid_test_df) // 2
        valid_df = valid_test_df[:half_valid_idx]
        test_df = valid_test_df[half_valid_idx:]
        train_data['text'] = train_df['text'].tolist()
        train_data['label'] = train_df['toxic'].tolist()
        valid_data['text'] = valid_df['text'].tolist()
        valid_data['label'] = valid_df['toxic'].tolist()
        test_data['text'] = test_df['text'].tolist()
        test_data['label'] = test_df['toxic'].tolist()
        
    
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
        for idx in tqdm(range(len(split_data['text'])), desc=f'Preprocessing {split} data', position=0, leave=True):
            text = split_data['text'][idx]
            label = split_data['label'][idx]
            if args.model_type == 'kcbert':
                args.max_seq_len = 300
            clean_text = bs4.BeautifulSoup(text, 'lxml').text
            clean_text = clean_text.replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')
            clean_text = ' '.join(clean_text.split())

            tokenized = tokenizer(clean_text, padding='max_length', truncation=True,
                                  max_length=args.max_seq_len, return_tensors='pt')

            data_dict[split]['input_ids'].append(tokenized['input_ids'].squeeze())
            data_dict[split]['attention_mask'].append(tokenized['attention_mask'].squeeze())
            if args.model_type in ['bert', 'albert', 'electra', 'deberta', 'debertav3','bert-large', 'roberta-large']:
                data_dict[split]['token_type_ids'].append(tokenized['token_type_ids'].squeeze())
            else: 
                data_dict[split]['token_type_ids'].append(torch.zeros(args.max_seq_len, dtype=torch.long))
            data_dict[split]['labels'].append(torch.tensor(label, dtype=torch.long)) 

        with open(os.path.join(preprocessed_path, f'{split}_processed.pkl'), 'wb') as f:
            pickle.dump(data_dict[split], f)
