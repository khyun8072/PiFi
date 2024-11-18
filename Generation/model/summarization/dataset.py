# Standard Library Modules
import os
import sys
import pickle
import torch
from tqdm.auto import tqdm
from torch.utils.data.dataset import Dataset
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

class SummarizationDataset(Dataset):
    def __init__(self, args, data_path: str, split: str) -> None:
        super(SummarizationDataset, self).__init__()
        self.args = args
        self.split = split
        with open(data_path, 'rb') as f:
            data_ = pickle.load(f)

        self.source_texts = data_['source_text']
        self.target_texts = data_['target_text']
        self.source_attention_masks = data_['source_attention_mask']
        self.target_attention_masks = data_['target_attention_mask']

    def __getitem__(self, idx: int) -> dict:
        source_text = self.source_texts[idx]
        target_text = self.target_texts[idx]
        source_attention_mask = self.source_attention_masks[idx]
        target_attention_mask = self.target_attention_masks[idx]

        return {
            'source_text': source_text,
            'target_text': target_text,
            'source_attention_mask': source_attention_mask,
            'target_attention_mask': target_attention_mask,
        }

    def __len__(self) -> int:
        return len(self.source_texts)

def collate_fn(data):
    source_texts = torch.stack([sample['source_text'] for sample in data])  
    target_texts = torch.stack([sample['target_text'] for sample in data])  
    source_attention_masks = torch.stack([sample['source_attention_mask'] for sample in data])  
    target_attention_masks = torch.stack([sample['target_attention_mask'] for sample in data])  

    return {
        'source_texts': source_texts,
        'target_texts': target_texts,
        'source_attention_masks': source_attention_masks,
        'target_attention_masks': target_attention_masks,
    }