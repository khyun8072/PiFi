import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  
import sys
import logging
import argparse
from tqdm.auto import tqdm
import json
import pandas as pd
import torch
torch.set_num_threads(2)  
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from nlgeval import NLGEval
from bert_score import BERTScorer
from utils.bart_score import BARTScorer  
from rouge_score import rouge_scorer

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model.summarization.model import SummarizationModel
from model.summarization.dataset import SummarizationDataset, collate_fn
from utils.utils import TqdmLoggingHandler, write_log, get_tb_exp_name, get_wandb_exp_name, get_torch_device, get_huggingface_model_name, check_path
from transformers import AutoTokenizer
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

    write_log(logger, "Loading dataset...")
    dataset_test = SummarizationDataset(args, os.path.join(args.preprocess_path, args.task, args.task_dataset, args.model_type, f'test.pkl'), 'tset')
    dataloader_test = DataLoader(dataset_test, batch_size=args.test_batch_size, num_workers=args.num_workers,
                                 shuffle=False, pin_memory=True, drop_last=False, collate_fn=collate_fn)

    write_log(logger, 'Loaded data successfully')
    write_log(logger, f'Test dataset size / iterations: {len(dataset_test)} / {len(dataloader_test)}')

    write_log(logger, 'Building model')
    model = SummarizationModel(args).to(device)
    model_name = get_huggingface_model_name(args.model_type)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    write_log(logger, "Loading model weights")
    load_model_name = os.path.join(args.model_path, args.task, args.task_dataset, args.model_type, args.method, args.llm_model, str(args.layer_num), 'final_model.pt')
    model = model.to('cpu')
    checkpoint = torch.load(load_model_name, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    write_log(logger, f'Loaded model weights from {load_model_name}')
    del checkpoint

    if args.use_wandb:
        import wandb
        wandb.init(project=args.proj_name,
                   name=get_wandb_exp_name(args) + f' - Test',
                   config=args,
                   notes=args.description,
                   tags=["TEST",
                         f"Dataset: {args.task_dataset}",
                         f"Model: {args.model_type}",
                         f"Method: {args.method}",
                         f"LLM: {args.llm_model}",
                         f"LLM_Layer: {args.layer_num}"])
    cls_loss = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing_eps)
    write_log(logger, f'Loss function: {cls_loss}')

    model = model.eval()
    ref_list = []
    hyp_list = []
    for test_iter_idx, data_dicts in enumerate(tqdm(dataloader_test, total=len(dataloader_test), desc='Testing', position=0, leave=True)):
        src = data_dicts['source_texts'].to(device)
        tgt = data_dicts['target_texts'].to(device)
        src_attention_mask = data_dicts['source_attention_masks'].to(device)
        tgt_attention_mask = data_dicts['target_attention_masks'].to(device)

        with torch.no_grad():
            generated_tokens = model.generate(src, src_attention_mask, args.max_seq_len)
        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(tgt, skip_special_tokens=True)
        each_reference = [each_ref.replace(' .', '.') for each_ref in decoded_labels]
        ref_list.append(each_reference)
        hyp_list.append(decoded_preds[0])
    write_log(logger, "TEST - Calculating NLG-eval metrics...")
    Eval = NLGEval(metrics_to_omit=['CIDEr', 'SPICE', 'SkipThoughtCS', 'EmbeddingAverageCosineSimilairty', 'VectorExtremaCosineSimilarity', 'GreedyMatchingScore'])
    BERT_Eval = BERTScorer(device=device, model_type='bert-base-multilingual-cased')
    BART_Eval = BARTScorer(device=device, checkpoint='facebook/bart-large-cnn')
    ROUGE_Eval = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
    _strip = lambda x: x.strip()
    ref_list2 = [list(map(_strip, refs)) for refs in zip(*ref_list)]
    metrics_dict = Eval.compute_metrics(ref_list2, hyp_list)

    bert_score_P, bert_score_R, bert_score_F1, bart_score_total = 0, 0, 0, 0
    rouge1_total, rouge2_total, rougeL_total, rougeLsum_total = 0, 0, 0, 0

    for each_ref, each_hyp in tqdm(zip(ref_list2[0], hyp_list), total=len(ref_list2[0]), desc=f'TEST - Calculating BERTScore, BARTScore, ROUGE...'):
        P, R, F1 = BERT_Eval.score([each_ref], [each_hyp])
        bert_score_P += P.item()
        bert_score_R += R.item()
        bert_score_F1 += F1.item()
        
        bart_score = BART_Eval.multi_ref_score([each_ref], [each_hyp], agg='max')
        bart_score_total += bart_score[0].item()

        rouge_scores = ROUGE_Eval.score(each_ref, each_hyp)
        rouge1_total += rouge_scores['rouge1'].fmeasure
        rouge2_total += rouge_scores['rouge2'].fmeasure
        rougeL_total += rouge_scores['rougeL'].fmeasure
        rougeLsum_total += rouge_scores['rougeLsum'].fmeasure

    bert_score_P /= len(ref_list2[0])
    bert_score_R /= len(ref_list2[0])
    bert_score_F1 /= len(ref_list2[0])
    bart_score_total /= len(ref_list2[0])
    rouge1_total /= len(ref_list2[0])
    rouge2_total /= len(ref_list2[0])
    rougeL_total /= len(ref_list2[0])
    rougeLsum_total /= len(ref_list2[0])

    write_log(logger, f"TEST - Bleu_1: {metrics_dict['Bleu_1']:.4f}")
    write_log(logger, f"TEST - Bleu_2: {metrics_dict['Bleu_2']:.4f}")
    write_log(logger, f"TEST - Bleu_3: {metrics_dict['Bleu_3']:.4f}")
    write_log(logger, f"TEST - Bleu_4: {metrics_dict['Bleu_4']:.4f}")
    bleu_avg = (metrics_dict['Bleu_1'] + metrics_dict['Bleu_2'] + metrics_dict['Bleu_3'] + metrics_dict['Bleu_4']) / 4
    write_log(logger, f"TEST - Bleu_avg: {bleu_avg:.4f}")
    write_log(logger, f"TEST - Rouge_1: {rouge1_total:.4f}")
    write_log(logger, f"TEST - Rouge_2: {rouge2_total:.4f}")
    write_log(logger, f"TEST - Rouge_L: {rougeL_total:.4f}")
    write_log(logger, f"TEST - Rouge_Lsum: {rougeLsum_total:.4f}")
    write_log(logger, f"TEST - Rouge_L_NLGEVAL: {metrics_dict['ROUGE_L']:.4f}")
    write_log(logger, f"TEST - Meteor: {metrics_dict['METEOR']:.4f}")
    write_log(logger, f"TEST - BERTScore_Precision: {bert_score_P:.4f}")
    write_log(logger, f"TEST - BERTScore_Recall: {bert_score_R:.4f}")
    write_log(logger, f"TEST - BERTScore_F1: {bert_score_F1:.4f}")
    write_log(logger, f"TEST - BARTScore: {bart_score_total:.4f}")

    save_path = os.path.join(args.result_path, args.task, args.task_dataset)
    check_path(save_path)  

    result_dict = {
        'args': vars(args),
        'Bleu_1': metrics_dict['Bleu_1'],
        'Bleu_2': metrics_dict['Bleu_2'],
        'Bleu_3': metrics_dict['Bleu_3'],
        'Bleu_4': metrics_dict['Bleu_4'],
        'Bleu_avg': bleu_avg,
        'Rouge_1': rouge1_total,
        'Rouge_2': rouge2_total,
        'Rouge_L': rougeL_total,
        'Rouge_Lsum': rougeLsum_total,
        'Rouge_L_NLG': metrics_dict['ROUGE_L'],
        'Meteor': metrics_dict['METEOR'],
        'BERTScore_Precision': bert_score_P,
        'BERTScore_Recall': bert_score_R,
        'BERTScore_F1': bert_score_F1,
        'BARTScore': bart_score_total,
    }

    save_name = os.path.join(save_path, f'test_result_{args.model_type}.json')
    with open(save_name, 'w') as f:
        json.dump(result_dict, f, indent=4, ensure_ascii=False)

    # Log the results to wandb if enabled
    if args.use_wandb:
        wandb_df = pd.DataFrame({
            'Dataset': [args.task_dataset],
            'Model': [args.model_type],
            'Bleu_1': [metrics_dict['Bleu_1']],
            'Bleu_2': [metrics_dict['Bleu_2']],
            'Bleu_3': [metrics_dict['Bleu_3']],
            'Bleu_4': [metrics_dict['Bleu_4']],
            'Bleu_avg': [bleu_avg],
            'Rouge_1': [rouge1_total],
            'Rouge_2': [rouge2_total],
            'Rouge_L': [rougeL_total],
            'Rouge_Lsum': [rougeLsum_total],
            'Rouge_L_NLG': [metrics_dict['ROUGE_L']],
            'Meteor': [metrics_dict['METEOR']],
            'BERTScore_Precision': [bert_score_P],
            'BERTScore_Recall': [bert_score_R],
            'BERTScore_F1': [bert_score_F1],
            'BARTScore': [bart_score_total],
        })
        
        wandb_table = wandb.Table(dataframe=wandb_df)
        wandb.log({"TEST_Result": wandb_table})
        wandb.save(save_name)
        wandb.finish()

    return metrics_dict
