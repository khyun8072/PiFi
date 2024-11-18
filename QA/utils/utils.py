# Standard Library Modules
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
    """
    Check if the path exists and create it if not.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def set_random_seed(seed: int):
    """
    Set random seed for reproducibility.
    """
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
            print("No GPU found. Using CPU.")
            return torch.device('cpu')
    elif 'mps' in device: 
        if not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install"
                      " was not built with MPS enabled.")
                print("Using CPU.")
            else:
                print("MPS not available because the current MacOS version"
                      " is not 12.3+ and/or you do not have an MPS-enabled"
                      " device on this machine.")
                print("Using CPU.")
            return torch.device('cpu')
        else:
            return torch.device(get_torch_device.device)
    elif 'cpu' in device:
        return torch.device('cpu')
    else:
        print("No such device found. Using CPU.")
        return torch.device('cpu')

class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.DEBUG):
        super().__init__(level)
        self.stream = sys.stdout

    def flush(self):
        self.acquire()
        try:
            if self.stream and hasattr(self.stream, "flush"):
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
    Get the experiment name for tensorboard experiment.
    """

    ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.localtime())

    exp_name = str()
    exp_name += "%s - " % args.task.upper()
    exp_name += "%s - " % args.proj_name

    if args.job in ['training', 'resume_training']:
        exp_name += 'TRAIN - '
        exp_name += "MODEL=%s - " % args.model_type.upper()
        exp_name += "DATA=%s - " % args.task_dataset.upper()
        exp_name += "DESC=%s - " % args.description
    elif args.job == 'testing':
        exp_name += 'TEST - '
        exp_name += "MODEL=%s - " % args.model_type.upper()
        exp_name += "DATA=%s - " % args.task_dataset.upper()
        exp_name += "DESC=%s - " % args.description
    exp_name += "TS=%s" % ts

    return exp_name

def get_wandb_exp_name(args: argparse.Namespace):
    """
    Get the experiment name for weight and biases experiment.
    """

    exp_name = str(args.seed)
    exp_name += "%s - " % args.task.upper()
    exp_name += "%s / " % args.task_dataset.upper()
    if args.task_dataset != args.test_dataset:
        exp_name += "%s / " % args.test_dataset
    exp_name += "%s" % args.model_type.upper()
    exp_name += " - %s" % args.method.upper()
    if 'ablation' in args.proj_name:
        exp_name += ' - %s' % args.padding.upper()
    if args.method == 'base_llm':
        if args.freeze == False:
            exp_name += " - freeze(x)"
        exp_name += " - %s" % args.llm_model.upper()
        exp_name += "(%s)" % args.layer_num
    
        

    return exp_name

def get_huggingface_model_name(model_type: str) -> str:
    name = model_type.lower()

    if name in ['bert', 'cnn', 'lstm', 'gru', 'rnn', 'transformer_enc']:
        return 'google-bert/bert-base-uncased'
    elif name == 'bert_large':
        return 'google-bert/bert-large-uncased'
    elif name == 'bart':
        return 'facebook/bart-large'
    elif name == 't5':
        return 't5-large'
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
    elif name == 'qwen2_72b':
        return 'Qwen/Qwen2-72B'
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

def llm_layer(llm_model_name, args):
    llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name, cache_dir=args.cache_path)

    if args.freeze == True:
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