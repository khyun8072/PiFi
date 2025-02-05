import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW
from datasets import load_dataset
from tqdm import tqdm
import argparse
from sklearn.metrics import accuracy_score, f1_score
import os
import re
import pickle

# Argument parser
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="HuggingFaceTB/SmolLM2-360M", help='Model name')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (SST2, IMDB, TweetSentimentBinary, TweetOffensive, CoLA)')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--max_length', type=int, default=128, help='Max sequence length for tokenization')
    parser.add_argument('--output_dir', type=str, default='/nas_homes/kyeonghyun', help='Directory to save models and results')
    parser.add_argument('--generate_max_length', type=int, default=200, help='Max length for generation during testing')
    parser.add_argument('--padding_side', type=str, default='right', help='Padding side for tokenization (right or left)')
    return parser.parse_args()

# Text Classification Dataset
import torch
from torch.utils.data import Dataset

class TextClassificationDataset(Dataset):
    def __init__(self, dataset, tokenizer, task_name, max_length, padding_side, is_validation):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.task_name = task_name
        self.max_length = max_length
        self.is_validation = is_validation
        self.padding_side = padding_side  # Padding side 설정 확인

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]

        # Task-specific prompts and labels
        if self.task_name == "SST2":
            label = "Positive" if example["label"] == 1 else "Negative"
            prompt = f"Sentence: {example['sentence']} \nSentiment:"
        elif self.task_name == "IMDB":
            label = "Positive" if example["label"] == 1 else "Negative"
            prompt = f"Review: {example['text']} \nSentiment:"
        elif self.task_name == "TweetSentimentBinary":
            label = "Positive" if example["label"] == 1 else "Negative"
            prompt = f"Tweet: {example['text']} \nSentiment:"
        elif self.task_name == "TweetOffensive":
            label = "Offensive" if example["label"] == 1 else "Non-Offensive"
            prompt = f"Tweet: {example['text']} \nOffensive:"
        elif self.task_name == "CoLA":
            label = "Acceptable" if example["label"] == 1 else "Unacceptable"
            prompt = f"Sentence: {example['sentence']} \nAcceptability:"
        else:
            raise ValueError(f"Unsupported task: {self.task_name}")

        if self.is_validation:
            # Tokenize the prompt only
            tokenized = self.tokenizer(
                prompt,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            )
            return {
                "input_ids": tokenized['input_ids'].squeeze(0),
                "attention_mask": tokenized['attention_mask'].squeeze(0),
                "prompt": prompt,
                "label_text": label
            }
        else:
            # Combine sentence and label
            combined_text = f"{prompt} {label}{self.tokenizer.eos_token}"

            # Tokenize the combined text
            tokenized = self.tokenizer(
                combined_text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            )

            # Mask the input portion in labels with -100 for loss calculation
            labels = tokenized['input_ids'].clone()

            # Calculate prompt and label lengths
            prompt_length = len(self.tokenizer(prompt)['input_ids'])
            label_length = len(self.tokenizer(label)['input_ids'])

            if self.padding_side == "right":
                # 오른쪽 패딩일 경우
                labels[:, :prompt_length] = -100
                labels[:, (prompt_length+label_length):] = -100
                
            elif self.padding_side == "left":
                # 왼쪽 패딩일 경우
                labels = tokenized['input_ids'].clone()
                lable_length = len(self.tokenizer(label)['input_ids'])
                labels[:, :-(lable_length-1)] = -100
            else:
                raise ValueError(f"Unsupported padding_side: {self.padding_side}")

            return {
                "input_ids": tokenized['input_ids'].squeeze(0),
                "attention_mask": tokenized['attention_mask'].squeeze(0),
                "labels": labels.squeeze(0),
                "prompt": prompt,
                "label_text": label
            }
        
    

def collate_fn(batch):
    return {
        key: torch.stack([b[key] for b in batch]) if isinstance(batch[0][key], torch.Tensor) else [b[key] for b in batch]
        for key in batch[0]
    }
import re

def extract_first_answer(generated_text):
    # 가장 마지막 ":" 이후의 텍스트 추출
    match = re.search(r":\s*([^\n]+)$", generated_text)
    if match:
        answer = match.group(1).strip()
        return answer.split()[0]  # 첫 번째 단어만 반환
    return None

# Validation function 수정
def validate_model(model, val_loader, tokenizer, device, generate_max_length):
    model.eval()
    all_prompts, all_generated, all_preds, all_labels = [], [], [], []
    label_mapping = {
        "Positive": 1,
        "Negative": 0,
        "Offensive": 1,
        "Non-Offensive": 0,
        "Acceptable": 1,
        "Unacceptable": 0
    }
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            prompts = batch['prompt']  # Save the original prompts
            labels = batch['label_text']  # Save the text labels

            # Generate predictions
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=generate_max_length,
                num_beams=5,
                early_stopping=True,
                eos_token_id=tokenizer.eos_token_id
            )

            # Decode generated text
            generated_texts = [tokenizer.decode(output, skip_special_tokens=True).strip() for output in outputs]
            preds = [extract_first_answer(gen) for gen in generated_texts]
            # Collect results
            all_prompts.extend(prompts)
            
            all_generated.extend(generated_texts)
            all_preds.extend(preds)
            all_labels.extend(labels)

    # Convert text labels to numerical
    numerical_labels = [label_mapping[label] for label in all_labels]
    numerical_generated = [label_mapping.get(gen, -1) for gen in all_generated]

    # Calculate metrics
    accuracy = accuracy_score(numerical_labels, numerical_generated)
    f1 = f1_score(numerical_labels, numerical_generated, average='weighted')

    return all_prompts, all_generated, all_preds, all_labels, accuracy, f1

# Training function
def train_model(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    count = 0
    for batch in tqdm(train_loader, desc="Training"):
      
        inputs = {k: v.to(device) for k, v in batch.items() if k not in ['prompt', 'label_text']}
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"]
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        count+=1
        if count % 100 == 0:
            print(f"Iter {count}: Loss = {loss.item():.4f}")
    return total_loss / len(train_loader)

# Ensure directory exists
def ensure_dir(directory):
    os.makedirs(directory, exist_ok=True)
    
# Split train into train and validation
def split_dataset(dataset, split_ratio=0.1):
    train_size = int((1 - split_ratio) * len(dataset))
    val_size = len(dataset) - train_size
    return torch.utils.data.random_split(dataset, [train_size, val_size])

# Save validation or test results
# Save validation or test results with performance metrics
def save_results_as_pickle(prompts, generated, preds, actual, output_file, metrics):
    """
    Save the prompts, generated outputs, actual labels, and performance metrics to a pickle file.

    Args:
        prompts (list): List of input prompts.
        generated (list): List of generated outputs.
        actual (list): List of actual labels.
        output_file (str): Path to the output pickle file.
        metrics (dict): Performance metrics (accuracy, f1).
    """
    results = {
        "prompts": prompts,
        "generated": generated,
        "predictions": preds,
        "actual_labels": actual,
        "metrics": metrics
    }

    with open(output_file, 'wb') as f:
        pickle.dump(results, f)

    print(f"Results and metrics saved to {output_file} (pickle format)")

# Main function
def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load tokenizer and model
    if 'OpenELM' in args.model_name:
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # 또는 tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.padding_side = args.padding_side
    if 'OpenELM' in args.model_name:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)

    # Load dataset
    datasets_config = {
        "SST2": ("glue", "sst2"),
        "IMDB": ("imdb", None),
        "TweetSentimentBinary": ("tweet_eval", "sentiment"),
        "TweetOffensive": ("tweet_eval", "offensive"),
        "CoLA": ("glue", "cola"),
    }
    if args.dataset not in datasets_config:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    dataset = load_dataset(*datasets_config[args.dataset])

    # Split dataset
    if 'validation' in dataset:
        train_data = TextClassificationDataset(dataset['train'], tokenizer, args.dataset, args.max_length, args.padding_side, is_validation=False)
        val_data = TextClassificationDataset(dataset['validation'], tokenizer, args.dataset, args.max_length, args.padding_side, is_validation=True)
    else:
        train_split, val_split = split_dataset(dataset['train'])
        val_data = TextClassificationDataset(val_split, tokenizer, args.dataset, args.max_length, args.padding_side, is_validation=True)
        train_data = TextClassificationDataset(train_split, tokenizer, args.dataset, args.max_length, args.padding_side, is_validation=False)
    test_data = TextClassificationDataset(dataset['test'], tokenizer, args.dataset, args.max_length, args.padding_side, is_validation=True)

    # Ensure directories exist
    val_results_dir = os.path.join(args.output_dir, "val_results")
    model_dir = os.path.join(args.output_dir, "model_final")
    test_results_dir = os.path.join(args.output_dir, "results")
    ensure_dir(val_results_dir)
    ensure_dir(model_dir)
    ensure_dir(test_results_dir)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, collate_fn=collate_fn)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    for i, batch in enumerate(train_loader):
        inputs = {k: v.to(device) for k, v in batch.items() if k not in ['prompt', 'label_text']}
        print(f"Input IDs: {inputs['input_ids'][0]}")
        print(f"Labels: {inputs['labels'][0]}")
        print(f"Decoded Input: {tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)}")
        # print(f"Decoded Label: {tokenizer.decode(inputs['labels'][0], skip_special_tokens=True)}")
        break

    # Train and validate
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        train_loss = train_model(model, train_loader, optimizer, device)
        print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}")

        # Validation step
        val_prompts, val_generated, val_preds, val_labels, val_accuracy, val_f1 = validate_model(
            model, val_loader, tokenizer, device, args.generate_max_length
        )
        val_results_path = os.path.join(val_results_dir, f"{args.model_name.replace('/', '_')}_{args.dataset}_val_epoch_{epoch + 1}.pkl")
        save_results_as_pickle(
        val_prompts,
        val_generated,
        val_preds,
        val_labels,
        val_results_path,
        {"accuracy": val_accuracy, "f1_score": val_f1}
        )
        print(f"Validation Accuracy: {val_accuracy:.4f}, F1 Score: {val_f1:.4f}")

    # Test step
    test_prompts, test_generated, test_preds,test_labels, test_accuracy, test_f1 = validate_model(
        model, test_loader, tokenizer, device, args.generate_max_length
    )
    test_results_path = os.path.join(test_results_dir, f"{args.model_name.replace('/', '_')}_{args.dataset}_test_results.pkl")
    save_results_as_pickle(
        test_prompts,
        test_generated,
        test_preds,
        test_labels,
        test_results_path,
        {"accuracy": test_accuracy, "f1_score": test_f1}
    )
    print(f"Test Accuracy: {test_accuracy:.4f}, F1 Score: {test_f1:.4f}")

if __name__ == "__main__":
    main()