import os
import argparse
from utils.utils import parse_bool

class ArgParser():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.proj_name = 'bert_llama'
        self.user_name = os.getlogin()


        # Task arguments
        task_list = ['question_answering']
        self.parser.add_argument('--task', type=str, choices=task_list, default='question_answering',
                                 help='Task to do; Must be given.')
        job_list = ['preprocessing', 'training', 'resume_training', 'testing']
        self.parser.add_argument('--job', type=str, choices=job_list, default='training',
                                 help='Job to do; Must be given.')
        dataset_list = ['squad']
        self.parser.add_argument('--task_dataset', type=str, choices=dataset_list, default='sst2',
                                 help='Dataset for the task; Must be given.')
        
        self.parser.add_argument('--description', type=str, default='default',
                                 help='Description of the experiment; Default is "default"')
        method_list = ['base', 'base_llm']
        self.parser.add_argument('--method', type=str, choices=method_list, default='base_llm',
                                 help='Method to use; Default is "base_llm"')


        # Path arguments - Modify these paths to fit your environment
        self.parser.add_argument('--data_path', type = str, default = f'/nas_homes/{self.user_name}/dataset/',
                                 help = 'Path to the raw dataset before preprocessing')
        self.parser.add_argument('--cache_path', type = str, default = f'/nas_homes/{self.user_name}/model',
                                 help = 'Path to the cache file.')
        self.parser.add_argument('--preprocess_path', type = str, default = f'/nas_homes/{self.user_name}/preprocessed/{self.proj_name}',
                                 help = 'Path to the preprocessed dataset.')
        self.parser.add_argument('--model_path', type=str, default=f'/nas_homes/{self.user_name}/model_final/{self.proj_name}',
                                 help='Path to the model after training.')
        self.parser.add_argument('--checkpoint_path', type=str, default=f'/nas_homes/{self.user_name}/model_checkpoint/{self.proj_name}')
        self.parser.add_argument('--result_path', type = str, default = f'/nas_homes/{self.user_name}/results/{self.proj_name}',
                                 help = 'Path to the result after testing.')
        self.parser.add_argument('--log_path', type = str, default = f'/nas_homes/{self.user_name}/tensorbord_log/{self.proj_name}',
                                 help = 'Path to the tensorboard log file.')

        # Model - Basic arguments
        self.parser.add_argument('--proj_name', type=str, default='Bert_llama',
                                 help='Name of the project.')
        model_type_list = ['bert', 'bert_large', 'roberta', 'albert', 'electra', 'deberta', 'debertav3', 'bert-large', 'roberta-large', 'kcbert', 'mbert']
        self.parser.add_argument('--model_type', type=str, choices=model_type_list, default='bert',
                                 help='Type of the classification model to use.')
        llm_model_list = ['llama2', 'llama3','llama3.1','llama3.1_instruct', 'mistral0.1', 'mistral0.3', 'qwen2_7b', 'qwen2_0.5b', 'qwen2_1.5b', 'qwen2_72b', 'gemma2', 'falcon', 'kollama', 'gerllama', 'chillama']
        self.parser.add_argument('--llm_model', type=str, choices=llm_model_list, default='llama3',
                                    help='LLM model to use; Default is llama3')
        self.parser.add_argument('--model_ispretrained', type=parse_bool, default=True,
                                 help='Whether to use pretrained model; Default is True')
        self.parser.add_argument('--rnn_isbidirectional', type=parse_bool, default=True,
                                 help='Whether to use bidirectional RNNs; Default is True')
        self.parser.add_argument('--min_seq_len', type=int, default=4,
                                 help='Minimum sequence length of the input; Default is 4')
        self.parser.add_argument('--max_seq_len', type=int, default=100,
                                 help='Maximum sequence length of the input; Default is 100')
        self.parser.add_argument('--dropout_rate', type=float, default=0.2,
                                 help='Dropout rate of the model; Default is 0.2')
        self.parser.add_argument('--padding', type=str, default='cls',
                                 help='Padding method of the input; Default is "cls"')
        self.parser.add_argument('--freeze', type=parse_bool, default=True,
                                 help='Freeze certain layers of the model during training; Default is True')

        # Model - Size arguments
        self.parser.add_argument('--embed_size', type=int, default=768, # Will be automatically specified by the model type if model is PLM
                                 help='Embedding size of the model; Default is 768')
        self.parser.add_argument('--hidden_size', type=int, default=768, # Will be automatically specified by the model type if model is PLM
                                 help='Hidden size of the model; Default is 768')
        self.parser.add_argument('--num_layers_rnn', type=int, default=2,
                                 help='Number of layers of RNNs; Default is 2')
        self.parser.add_argument('--num_layers_transformer', type=int, default=6,
                                 help='Number of layers of Transformer Encoder; Default is 6')
        self.parser.add_argument('--num_heads_transformer', type=int, default=8,
                                 help='Number of heads of Transformer Encoder; Default is 8')
        self.parser.add_argument('--layer_num', type=int, default=-1,
                                 help='Layer number of the LLM model; Default is -1')

        # Model - Optimizer & Scheduler arguments
        optim_list = ['SGD', 'AdaDelta', 'Adam', 'AdamW']
        scheduler_list = ['None', 'StepLR', 'LambdaLR', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts', 'ReduceLROnPlateau']
        self.parser.add_argument('--optimizer', type=str, choices=optim_list, default='Adam',
                                 help="Optimizer to use; Default is Adam")
        self.parser.add_argument('--scheduler', type=str, choices=scheduler_list, default='None',
                                 help="Scheduler to use for classification; If None, no scheduler is used; Default is None")

        # Training arguments 1
        self.parser.add_argument('--num_epochs', type=int, default=3,
                                 help='Training epochs; Default is 50')
        self.parser.add_argument('--learning_rate', type=float, default=5e-5,
                                 help='Learning rate of optimizer; Default is 5e-5')
        # Training arguments 2
        self.parser.add_argument('--num_workers', type=int, default=2,
                                 help='Num CPU Workers; Default is 2')
        self.parser.add_argument('--batch_size', type=int, default=32,
                                 help='Batch size; Default is 32')
        self.parser.add_argument('--weight_decay', type=float, default=0,
                                 help='Weight decay; Default is 5e-4; If 0, no weight decay')
        self.parser.add_argument('--clip_grad_norm', type=int, default=5,
                                 help='Gradient clipping norm; Default is 5')
        self.parser.add_argument('--label_smoothing_eps', type=float, default=0.05,
                                 help='Label smoothing epsilon; Default is 0.05')
        self.parser.add_argument('--early_stopping_patience', type=int, default=5,
                                 help='Early stopping patience; No early stopping if None; Default is 5')
        self.parser.add_argument('--train_valid_split', type=float, default=0.2,
                                 help='Train/Valid split ratio; Default is 0.2')
        objective_list = ['loss', 'accuracy', 'f1']
        self.parser.add_argument('--optimize_objective', type=str, choices=objective_list, default='accuracy',
                                 help='Objective to optimize; Default is accuracy')

        # Testing/Inference arguments
        self.parser.add_argument('--test_batch_size', default=16, type=int,
                                 help='Batch size for test; Default is 16')

        # Other arguments - Device, Seed, Logging, etc.
        self.parser.add_argument('--device', type=str, default='cuda',
                                 help='Device to use for training; Default is cuda')
        self.parser.add_argument('--seed', type=int, default=2023,
                                 help='Random seed; Default is 2023')
        self.parser.add_argument('--use_tensorboard', type=parse_bool, default=False,
                                 help='Using tensorboard; Default is True')
        self.parser.add_argument('--use_wandb', type=parse_bool, default=True,
                                 help='Using wandb; Default is True')
        self.parser.add_argument('--log_freq', default=500, type=int,
                                 help='Logging frequency; Default is 500')

    def get_args(self):
        return self.parser.parse_args()
