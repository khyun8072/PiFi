import pickle
import argparse
from tqdm.auto import tqdm
from torch.utils.data.dataset import Dataset


class CustomDataset(Dataset):
    def __init__(self, data_path: str) -> None:
        super(CustomDataset, self).__init__()
        with open(data_path, 'rb') as f:
            data_ = pickle.load(f)

        self.data_list = []
        self.src_vocab_size = data_['src_vocab_size']
        self.tgt_vocab_size = data_['tgt_vocab_size']
        for idx in tqdm(range(len(data_['EN_text_ids'])), desc = f'Loading data from {data_path}'):

            self.data_list.append({
                'EN_text_ids': data_['EN_text_ids'][idx],
                'DE_text_ids': data_['DE_text_ids'][idx],
                'src_attention_mask': data_['src_attention_mask'][idx],
                'tgt_attention_mask': data_['tgt_attention_mask'][idx],
                'index': idx
            })

        del data_
    def __getitem__(self, idx: int) -> dict:
        return self.data_list[idx]
    
    def __len__(self) -> int:
        return len(self.data_list)
    
