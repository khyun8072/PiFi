import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" 
import sys
import logging
import argparse
from tqdm.auto import tqdm
import pandas as pd
import torch
torch.set_num_threads(2)  
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model.question_answering.model import QAModel  
from model.question_answering.dataset import CustomDataset 
from utils.utils import TqdmLoggingHandler, write_log, get_tb_exp_name, get_wandb_exp_name, get_torch_device

def testing(args: argparse.Namespace) -> tuple: 
    device = get_torch_device(args.device)

    logger = logging.getLogger(__name__)
    if len(logger.handlers) > 0:
        logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(" %(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False

    if args.use_tensorboard:
        writer = SummaryWriter(os.path.join(args.log_path, get_tb_exp_name(args)))
        writer.add_text('args', str(args))

    write_log(logger, "Loading dataset")
    dataset_test = CustomDataset(os.path.join(args.preprocess_path, args.task, args.task_dataset, args.model_type, f'test_processed.pkl'))
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, num_workers=args.num_workers,
                                 shuffle=False, pin_memory=True, drop_last=False)
    args.vocab_size = dataset_test.vocab_size
    args.pad_token_id = dataset_test.pad_token_id

    write_log(logger, "Loaded data successfully")
    write_log(logger, f"Test dataset size / iterations: {len(dataset_test)} / {len(dataloader_test)}")

    write_log(logger, "Building model")
    model = QAModel(args).to(device)

    write_log(logger, "Loading model weights")
    load_model_name = os.path.join(args.model_path, args.task, args.task_dataset, args.padding, args.model_type, args.method, args.llm_model, str(args.layer_num), 'final_model.pt')
    model = model.to('cpu')
    checkpoint = torch.load(load_model_name, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    write_log(logger, f"Loaded model weights from {load_model_name}")
    del checkpoint

    if args.use_wandb:
        import wandb
        from wandb import AlertLevel
        wandb.init(project=args.proj_name,
                   name=get_wandb_exp_name(args) + f' - Test',
                   config=args,
                   notes=args.description,
                   tags=["TEST",
                         f"Dataset: {args.task_dataset}",
                         f"Model: {args.model_type}"])

    model = model.eval()
    test_loss = 0
    test_em = 0
    test_f1 = 0
    total = 0

    for test_iter_idx, data_dicts in enumerate(tqdm(dataloader_test, total=len(dataloader_test), desc="Testing", position=0, leave=True)):
        input_ids = data_dicts['input_ids'].to(device)
        attention_mask = data_dicts['attention_mask'].to(device)
        token_type_ids = data_dicts['token_type_ids'].to(device)
        start_positions = data_dicts['start_positions'].to(device)
        end_positions = data_dicts['end_positions'].to(device)

        with torch.no_grad():
            start_logits, end_logits = model(input_ids, attention_mask, token_type_ids)

        loss_fn = nn.CrossEntropyLoss()
        start_loss = loss_fn(start_logits, start_positions)
        end_loss = loss_fn(end_logits, end_positions)
        batch_loss = (start_loss + end_loss) / 2
        test_loss += batch_loss.item()

        start_preds = torch.argmax(start_logits, dim=1)
        end_preds = torch.argmax(end_logits, dim=1)

        for i in range(input_ids.size(0)):
            pred_start = start_preds[i].item()
            pred_end = end_preds[i].item()
            true_start = start_positions[i].item()
            true_end = end_positions[i].item()

            pred_span = set(range(pred_start, pred_end + 1))
            true_span = set(range(true_start, true_end + 1))

            em = int(pred_span == true_span)
            f1 = 2 * len(pred_span & true_span) / (len(pred_span) + len(true_span)) if len(pred_span | true_span) > 0 else 0

            test_em += em
            test_f1 += f1

        total += input_ids.size(0)

        if test_iter_idx % args.log_freq == 0 or test_iter_idx == len(dataloader_test) - 1:
            write_log(logger, f"TEST - Iter [{test_iter_idx}/{len(dataloader_test)}] - Loss: {batch_loss.item():.4f}")
            write_log(logger, f"TEST - Iter [{test_iter_idx}/{len(dataloader_test)}] - EM: {em:.4f}")
            write_log(logger, f"TEST - Iter [{test_iter_idx}/{len(dataloader_test)}] - F1: {f1:.4f}")

    # Test - Check loss, EM, and F1
    test_loss /= len(dataloader_test)
    test_em /= total
    test_f1 /= total

    # Final - End of testing
    write_log(logger, f"Done! - TEST - Loss: {test_loss:.4f} - EM: {test_em:.4f} - F1: {test_f1:.4f}")
    if args.use_tensorboard:
        writer.add_scalar('TEST/Loss', test_loss, 0)
        writer.add_scalar('TEST/EM', test_em, 0)
        writer.add_scalar('TEST/F1', test_f1, 0)
        writer.close()
    if args.use_wandb:
        wandb_df = pd.DataFrame({
            'Dataset': [args.task_dataset],
            'Model': [args.model_type],
            'EM': [test_em],
            'F1': [test_f1],
            'Loss': [test_loss]
        })
        wandb_table = wandb.Table(dataframe=wandb_df)
        wandb.log({'TEST_Result': wandb_table})
        wandb.finish()

    return test_em, test_f1
