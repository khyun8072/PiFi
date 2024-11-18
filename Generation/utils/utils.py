import os
import sys
import time
import tqdm
import random
import logging
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

def check_path(path: str):
    '''
    Check if the path exists and create it if not.
    '''
    if not os.path.exists(path):
        os.makedirs(path)

def set_random_seed(seed: int):
    '''
    Set random seed for repruducibility.
    '''
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_torch_device(device: str):
    if device is not None:
        get_torch_device.device = device
    
    if 'cuda' in get_torch_device.device: 
        if torch.cuda.is_available():
            return torch.device(get_torch_device.device) 
        else:
            print('No GPU found. Using CPU.')
            return torch.device('cpu')
    elif 'mps' in device: 
        if not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print('MPS not available because the current Pytorch install'
                      ' was not built with MPS enabled.')
                print('Using CPU.')
            else:
                print('MPS not available because the current MacOS version'
                      ' is not 12.3+ and/or you do not have an MPS-enabled'
                      ' device on this machine.')
                print('Using CPU.')
            return torch.device('cpu')
        else:
            return torch.device(get_torch_device.device)
    elif 'cpu' in device:
        return torch.device('cpu')
    else:
        print('No such device found. Using CPU.')
        return torch.device('cpu')

class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level = logging.DEBUG):
        super().__init__(level)
        self.stream = sys.stdout 

    def flush(self):
        self.acquire() 
        try:
            if self.stream and hasattr(self.stream, 'flush'): 
                self.stream.flush() 
 
        finally:
            self.release() 
    
    def emit(self, record):
        try:
            msg = self.format(record) 
            tqdm.tqdm.write(msg, self.stream) 
            self.flush()
        except (KeyboardInterrupt, SystemExit, RecursionError):
            raise
        except Exception:
            self.handleError(record)

def write_log(logger, message):
    if logger:
        logger.info(message) 

def get_tb_exp_name(args: argparse.Namespace):
    """
    tensorboard 실험을 위한 실험명 가져오기 
    """

    ts = time.strftime('%Y - %b - %d - %H: %M: %S', time.localtime())
    exp_name = str()
    exp_name += "%s - " % args.task.upper()
    exp_name += "%s - " % args.proj_name

    if args.job in ['training', 'resume_training']:
        exp_name += 'TRAIN - '
        exp_name += 'MODEL = %s - ' % args.model_type.upper()
        exp_name += 'DATA = %s - ' % args.task_dataset.upper()
        exp_name += 'DESC = %s - ' % args.description
    
    elif args.job == 'testing':
        exp_name += 'TEST - '
        exp_name += 'MODEL = %s - ' % args.model_type.upper()
        exp_name += 'DATA = %s - ' % args.task_dataset.upper()
        exp_name += 'DESC = %s - ' % args.description
    exp_name += 'TS = %s' % ts

    return exp_name


def get_wandb_exp_name(args: argparse.Namespace):
    """
    Get the experiment name for weight and biases experiment.
    """

    exp_name = str(args.seed)
    exp_name += "%s - " % args.task.upper()
    exp_name += "%s / " % args.task_dataset.upper()
    exp_name += "%s" % args.model_type.upper()
    exp_name += " - %s" % args.method.upper()
    if args.method == 'base_llm':
        exp_name += " - %s" % args.llm_model.upper()
        exp_name += "(%s)" % args.layer_num

    return exp_name

    return exp_name

def get_huggingface_model_name(model_type: str) -> str:
    name = model_type.lower()

def get_huggingface_model_name(model_type: str) -> str:
    name = model_type.lower()

    if name in ['bert', 'cnn', 'lstm', 'gru', 'rnn', 'transformer_enc']: 
        return 'google-bert/bert-base-uncased'
    elif name == 'bart':
        return 'facebook/bart-base'
    elif name == 't5':
        return 'google-t5/t5-base'
    elif name == 'roberta':
        return 'FacebookAI/roberta-base'
    elif name == 'electra':
        return 'google/electra-base-discriminator'
    elif name == 'albert':
        return 'albert-base-v2'
    elif name == 'deberta':
        return 'microsoft/deberta-base'
    elif name == 'debertav3':
        return 'microsoft/deberta-v3-base'
    elif name == 'mbert':
        return 'google-bert/bert-base-multilingual-cased'
    elif name == 'llama2':
        return 'meta-llama/Llama-2-7b-hf'
    elif name == 'llama3':
        return 'meta-llama/Meta-Llama-3-8B' 
    elif name == 'llama3.1':
        return 'meta-llama/Meta-Llama-3.1-8B'
    elif name == 'llama3.1_instruct':
        return 'meta-llama/Meta-Llama-3.1-8B-Instruct'
    elif name == 'mistral0.1':
        return 'mistralai/Mistral-7B-v0.1'
    elif name == 'mistral0.3':
        return 'mistralai/Mistral-7B-v0.3'
    elif name == 'qwen2_7b':
        return 'Qwen/Qwen2-7B'
    elif name == 'qwen2_0.5b':
        return 'Qwen/Qwen2-0.5B'
    elif name == 'qwen2_1.5b':
        return 'Qwen/Qwen2-1.5B'
    elif name == 'gemma2':
        return 'google/gemma-2-9b'
    elif name == 'falcon':
        return 'tiiuae/falcon-7b'
    elif name == 'kollama':
        return 'beomi/Llama-3-Open-Ko-8B'
    elif name == 'kcbert':
        return 'beomi/kcbert-base'
    elif name == 'gerllama':
        return 'DiscoResearch/Llama3-German-8B'
    elif name == 'chillama':
        return 'hfl/llama-3-chinese-8b'
    else:
        raise NotImplementedError
    
def parse_bool(value: str):
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def generate_square_subsequent_mask(size) -> torch.tensor:
    mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(args, src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=args.device).type(torch.bool)
    src_padding_mask = (src == args.PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == args.PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

def llm_layer(llm_model_name, args):
    llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name, cache_dir=args.cache_path)

    for param in llm_model.parameters():
        param.requires_grad = False
    if args.llm_model == 'falcon': 
        if len(llm_model.transformer.h) <= args.layer_num:
            raise ValueError(f"Layer {args.layer_num} does not exist in the model. Training halted.")
        else:
            llm_layer = llm_model.transformer.h[args.layer_num]
    else:
        if len(llm_model.model.layers) <= args.layer_num:
            raise ValueError(f"Layer {args.layer_num} does not exist in the model. Training halted.")
        else:
            llm_layer = llm_model.model.layers[args.layer_num]

    llm_embed_size = llm_model.config.hidden_size
    llm_hidden_size = llm_model.config.hidden_size
    return llm_layer, llm_embed_size, llm_hidden_size