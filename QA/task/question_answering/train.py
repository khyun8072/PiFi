import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" 
import sys
import shutil
import logging
import argparse
# 3rd-party Modules
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score
# Pytorch Modules
import torch
torch.set_num_threads(2) 
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model.question_answering.model import QAModel 
from model.question_answering.dataset import CustomDataset  
from model.optimizer.optimizer import get_optimizer
from model.optimizer.scheduler import get_scheduler
from utils.utils import TqdmLoggingHandler, write_log, get_tb_exp_name, get_wandb_exp_name, get_torch_device, check_path

def training(args: argparse.Namespace) -> None:
    device = get_torch_device(args.device)

    logger = logging.getLogger(__name__)
    if len(logger.handlers) > 0:
        logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(" %(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False

    write_log(logger, "Loading data")
    dataset_dict, dataloader_dict = {}, {}
    dataset_dict['train'] = CustomDataset(os.path.join(args.preprocess_path, args.task, args.task_dataset, args.model_type, f'train_processed.pkl'))
    dataset_dict['valid'] = CustomDataset(os.path.join(args.preprocess_path, args.task, args.task_dataset, args.model_type, f'valid_processed.pkl'))

    dataloader_dict['train'] = DataLoader(dataset_dict['train'], batch_size=args.batch_size, num_workers=args.num_workers,
                                          shuffle=True, pin_memory=True, drop_last=True)
    dataloader_dict['valid'] = DataLoader(dataset_dict['valid'], batch_size=args.batch_size, num_workers=args.num_workers,
                                          shuffle=False, pin_memory=True, drop_last=False)
    args.vocab_size = dataset_dict['train'].vocab_size
    args.pad_token_id = dataset_dict['train'].pad_token_id

    write_log(logger, "Loaded data successfully")
    write_log(logger, f"Train dataset size / iterations: {len(dataset_dict['train'])} / {len(dataloader_dict['train'])}")
    write_log(logger, f"Valid dataset size / iterations: {len(dataset_dict['valid'])} / {len(dataloader_dict['valid'])}")

    write_log(logger, "Building model")
    model = QAModel(args).to(device)

    write_log(logger, "Building optimizer and scheduler")
    optimizer = get_optimizer(model, learning_rate=args.learning_rate, weight_decay=args.weight_decay, optim_type=args.optimizer)
    scheduler = get_scheduler(optimizer, len(dataloader_dict['train']), num_epochs=args.num_epochs,
                              early_stopping_patience=args.early_stopping_patience, learning_rate=args.learning_rate,
                              scheduler_type=args.scheduler)
    write_log(logger, f"Optimizer: {optimizer}")
    write_log(logger, f"Scheduler: {scheduler}")

    loss_fn = nn.CrossEntropyLoss()
    write_log(logger, f"Loss function: {loss_fn}")
    write_log(logger, f"Method: {args.method}")

    start_epoch = 0
    if args.job == 'resume_training':
        write_log(logger, "Resuming training model")
        load_checkpoint_name = os.path.join(args.checkpoint_path, args.task, args.task_dataset, args.model_type,
                                            f'checkpoint.pt')
        model = model.to('cpu')
        checkpoint = torch.load(load_checkpoint_name, map_location='cpu')
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        model = model.to(device)
        write_log(logger, f"Loaded checkpoint from {load_checkpoint_name}")

        if args.use_wandb:
            import wandb 
            from wandb import AlertLevel
            wandb.init(project=args.proj_name,
                       name=get_wandb_exp_name(args),
                       config=args,
                       notes=args.description,
                       tags=["TRAIN",
                             f"Dataset: {args.task_dataset}",
                             f"Model: {args.model_type}",
                             f"Method: {args.method}",
                             f"LLM: {args.llm_model}",
                             f"LLM_Layer: {args.layer_num}"],
                       resume=True,
                       id=checkpoint['wandb_id'])
            wandb.watch(models=model, criterion=loss_fn, log='all', log_freq=10)
        del checkpoint

    if args.use_tensorboard:
        writer = SummaryWriter(os.path.join(args.log_path, get_tb_exp_name(args)))
        writer.add_text('args', str(args))

    if args.use_wandb and args.job == 'training':
        import wandb 
        from wandb import AlertLevel
        wandb.init(project=args.proj_name,
                       name=get_wandb_exp_name(args),
                       config=args,
                       notes=args.description,
                       tags=["TRAIN",
                             f"Dataset: {args.task_dataset}",
                             f"Model: {args.model_type}",
                             f"Method: {args.method}",
                             f"LLM: {args.llm_model}",
                             f"LLM_Layer: {args.layer_num}"])
        wandb.watch(models=model, criterion=loss_fn, log='all', log_freq=10)

    best_epoch_idx = 0
    best_valid_objective_value = None
    early_stopping_counter = 0

    write_log(logger, f"Start training from epoch {start_epoch}")
    for epoch_idx in range(start_epoch, args.num_epochs):
        model = model.train()
        train_loss = 0
        train_acc = 0

        for iter_idx, data_dicts in enumerate(tqdm(dataloader_dict['train'], total=len(dataloader_dict['train']), desc=f'Training - Epoch [{epoch_idx}/{args.num_epochs}]', position=0, leave=True)):
            input_ids = data_dicts['input_ids'].to(device)
            attention_mask = data_dicts['attention_mask'].to(device)
            token_type_ids = data_dicts['token_type_ids'].to(device)
            start_positions = data_dicts['start_positions'].to(device)
            end_positions = data_dicts['end_positions'].to(device)

            start_logits, end_logits = model(input_ids, attention_mask, token_type_ids)

            start_loss = loss_fn(start_logits, start_positions)
            end_loss = loss_fn(end_logits, end_positions)
            batch_loss = (start_loss + end_loss) / 2

            start_preds = torch.argmax(start_logits, dim=1)
            end_preds = torch.argmax(end_logits, dim=1)

            batch_acc = (torch.eq(start_preds, start_positions).float().mean().item() +
                         torch.eq(end_preds, end_positions).float().mean().item()) / 2

            optimizer.zero_grad()
            batch_loss.backward()
            if args.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
            if args.scheduler in ['StepLR', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts']:
                scheduler.step() 

            train_loss += batch_loss.item()
            train_acc += batch_acc

            if iter_idx % args.log_freq == 0 or iter_idx == len(dataloader_dict['train']) - 1:
                write_log(logger, f"TRAIN - Epoch [{epoch_idx}/{args.num_epochs}] - Iter [{iter_idx}/{len(dataloader_dict['train'])}] - Loss: {batch_loss.item():.4f}")
                write_log(logger, f"TRAIN - Epoch [{epoch_idx}/{args.num_epochs}] - Iter [{iter_idx}/{len(dataloader_dict['train'])}] - Acc: {batch_acc:.4f}")
            if args.use_tensorboard:
                writer.add_scalar('TRAIN/Learning_Rate', optimizer.param_groups[0]['lr'], epoch_idx * len(dataloader_dict['train']) + iter_idx)

        if args.use_tensorboard:
            writer.add_scalar('TRAIN/Loss', train_loss / len(dataloader_dict['train']), epoch_idx)
            writer.add_scalar('TRAIN/Acc', train_acc / len(dataloader_dict['train']), epoch_idx)

        model = model.eval()
        valid_loss = 0
        valid_acc = 0

        for iter_idx, data_dicts in enumerate(tqdm(dataloader_dict['valid'], total=len(dataloader_dict['valid']), desc=f'Validating - Epoch [{epoch_idx}/{args.num_epochs}]', position=0, leave=True)):
            input_ids = data_dicts['input_ids'].to(device)
            attention_mask = data_dicts['attention_mask'].to(device)
            token_type_ids = data_dicts['token_type_ids'].to(device)
            start_positions = data_dicts['start_positions'].to(device)
            end_positions = data_dicts['end_positions'].to(device)

            with torch.no_grad():
                start_logits, end_logits = model(input_ids, attention_mask, token_type_ids)

            start_loss = loss_fn(start_logits, start_positions)
            end_loss = loss_fn(end_logits, end_positions)
            batch_loss = (start_loss + end_loss) / 2

            start_preds = torch.argmax(start_logits, dim=1)
            end_preds = torch.argmax(end_logits, dim=1)

            batch_acc = (torch.eq(start_preds, start_positions).float().mean().item() +
                         torch.eq(end_preds, end_positions).float().mean().item()) / 2

            valid_loss += batch_loss.item()
            valid_acc += batch_acc

            if iter_idx % args.log_freq == 0 or iter_idx == len(dataloader_dict['valid']) - 1:
                write_log(logger, f"VALID - Epoch [{epoch_idx}/{args.num_epochs}] - Iter [{iter_idx}/{len(dataloader_dict['valid'])}] - Loss: {batch_loss.item():.4f}")
                write_log(logger, f"VALID - Epoch [{epoch_idx}/{args.num_epochs}] - Iter [{iter_idx}/{len(dataloader_dict['valid'])}] - Acc: {batch_acc:.4f}")

        if args.scheduler == 'LambdaLR':
            scheduler.step()
        elif args.scheduler == 'ReduceLROnPlateau':
            scheduler.step(valid_loss)

        valid_loss /= len(dataloader_dict['valid'])
        valid_acc /= len(dataloader_dict['valid'])

        valid_objective_value = valid_acc if args.optimize_objective == 'accuracy' else valid_loss * -1

        if best_valid_objective_value is None or valid_objective_value > best_valid_objective_value:
            best_valid_objective_value = valid_objective_value
            best_epoch_idx = epoch_idx
            write_log(logger, f"VALID - Saving checkpoint for best valid {args.optimize_objective}...")
            early_stopping_counter = 0 

            checkpoint_save_path = os.path.join(args.checkpoint_path, args.task, args.task_dataset, args.padding, args.model_type, args.method, args.llm_model, str(args.layer_num))
    
            check_path(checkpoint_save_path)

            torch.save({
                'epoch': epoch_idx,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler is not None else None
            }, os.path.join(checkpoint_save_path, f'checkpoint.pt'))
            write_log(logger, f"VALID - Best valid at epoch {best_epoch_idx} - {args.optimize_objective}: {abs(best_valid_objective_value):.4f}")
            write_log(logger, f"VALID - Saved checkpoint to {checkpoint_save_path}")
        else:
            early_stopping_counter += 1
            write_log(logger, f"VALID - Early stopping counter: {early_stopping_counter}/{args.early_stopping_patience}")

        if args.use_tensorboard:
            writer.add_scalar('VALID/Loss', valid_loss, epoch_idx)
            writer.add_scalar('VALID/Acc', valid_acc, epoch_idx)
        if args.use_wandb:
            wandb.log({'TRAIN/Epoch_Loss': train_loss / len(dataloader_dict['train']),
                       'TRAIN/Epoch_Acc': train_acc / len(dataloader_dict['train']),
                       'VALID/Epoch_Loss': valid_loss,
                       'VALID/Epoch_Acc': valid_acc,
                       'Epoch_Index': epoch_idx})
            wandb.alert(
                title='Epoch End',
                text=f"VALID - Epoch {epoch_idx} - Loss: {valid_loss:.4f} - Acc: {valid_acc:.4f}",
                level=AlertLevel.INFO,
                wait_duration=300
            )

        if early_stopping_counter >= args.early_stopping_patience:
            write_log(logger, f"VALID - Early stopping at epoch {epoch_idx}...")
            break

    write_log(logger, f"Done! Best valid at epoch {best_epoch_idx} - {args.optimize_objective}: {abs(best_valid_objective_value):.4f}")
    if args.use_tensorboard:
        writer.add_text('VALID/Best', f"Best valid at epoch {best_epoch_idx} - {args.optimize_objective}: {abs(best_valid_objective_value):.4f}")
        writer.close()

    final_model_save_path = os.path.join(args.model_path, args.task, args.task_dataset, args.padding, args.model_type, args.method, args.llm_model, str(args.layer_num))
    check_path(final_model_save_path)
    shutil.copyfile(os.path.join(checkpoint_save_path, 'checkpoint.pt'), os.path.join(final_model_save_path, 'final_model.pt')) # Copy best checkpoint as final model
    write_log(logger, f"FINAL - Saved final model to {final_model_save_path}")

    if args.use_wandb:
        wandb.finish()
