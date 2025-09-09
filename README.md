# PiFi: Plug-in and Fine-tuning

![pifi_figure](https://github.com/user-attachments/assets/e73cbce8-e680-419e-a883-13d05c5e2d98)

## Overview

PiFi is a novel methodology that bridges the gap between Small Language Models (SLMs) and Large Language Models (LLMs) by leveraging the frozen last layer of LLMs as a plug-in component for SLMs during fine-tuning. This approach enables SLMs to benefit from the rich representation capabilities of LLMs while maintaining computational efficiency.

**Paper**: [Plug-in and Fine-tuning: Bridging the Gap between Small Language Models and Large Language Models](https://aclanthology.org/2025.acl-long.271/)

## Methodology

PiFi works by:
1. **Freezing the Last Layer**: Taking the final layer from a pre-trained LLM and keeping it frozen
2. **Plug-in Architecture**: Integrating this frozen layer as a plug-in component into the SLM architecture
3. **Fine-tuning**: Training the SLM with the plugged-in LLM layer to improve performance on downstream tasks

This approach allows SLMs to leverage the knowledge encoded in LLM's final representations without the computational overhead of running the entire LLM during inference.

## Repository Structure

```
PiFi/
├── Classification/          # Classification task implementation
│   ├── main.py             # Main training/testing script
│   ├── run_classification.sh # Shell script for running experiments
│   ├── model/              # Model implementations
│   ├── task/               # Task-specific code
│   └── utils/              # Utility functions
├── TextualEntailment/      # Textual entailment task implementation
│   ├── main.py             # Main training/testing script  
│   ├── run_entailment.sh   # Shell script for running experiments
│   ├── model/              # Model implementations
│   ├── task/               # Task-specific code
│   └── utils/              # Utility functions
└── requirements.txt        # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/PiFi.git
cd PiFi
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Classification Tasks

The classification module supports multiple datasets: SST-2, IMDB, Tweet Sentiment Binary, Tweet Offensive, and CoLA.

#### Running Classification Experiments

Use the provided shell script to run all classification experiments:
```bash
cd Classification
bash run_classification.sh
```

Or run individual experiments:
```bash
# Preprocessing
python main.py --task classification --job=preprocessing --task_dataset=sst2 --model_type=bert

# Training baseline model
python main.py --task classification --job=training --task_dataset=sst2 --test_dataset=sst2 --model_type=bert --method=base

# Testing baseline model
python main.py --task classification --job=testing --task_dataset=sst2 --test_dataset=sst2 --model_type=bert --method=base

# Training with LLM plugin (PiFi)
python main.py --task classification --job=training --task_dataset=sst2 --test_dataset=sst2 --model_type=bert --method=base_llm --llm=llama3.1

# Testing with LLM plugin (PiFi)
python main.py --task classification --job=testing --task_dataset=sst2 --test_dataset=sst2 --model_type=bert --method=base_llm --llm=llama3.1
```

#### Available Parameters:
- `--task_dataset`: Dataset name (sst2, imdb, tweet_sentiment_binary, tweet_offensive, cola)
- `--model_type`: Base model type (bert)
- `--method`: Training method (base, base_llm)
- `--llm`: LLM to use for plugin (llama3.1)
- `--job`: Operation to perform (preprocessing, training, testing)

### Textual Entailment Tasks

The textual entailment module supports MNLI and SNLI datasets.

#### Running Entailment Experiments

Use the provided shell script to run all entailment experiments:
```bash
cd TextualEntailment
bash run_entailment.sh
```

Or run individual experiments:
```bash
# Preprocessing
python main.py --task entailment --job=preprocessing --task_dataset=mnli --model_type=bert

# Training baseline model
python main.py --task entailment --job=training --task_dataset=mnli --test_dataset=mnli --model_type=bert --method=base

# Testing baseline model  
python main.py --task entailment --job=testing --task_dataset=mnli --test_dataset=mnli --model_type=bert --method=base

# Training with LLM plugin (PiFi)
python main.py --task entailment --job=training --task_dataset=mnli --test_dataset=mnli --model_type=bert --method=base_llm --llm=llama3.1

# Testing with LLM plugin (PiFi)
python main.py --task entailment --job=testing --task_dataset=mnli --test_dataset=mnli --model_type=bert --method=base_llm --llm=llama3.1
```

#### Available Parameters:
- `--task_dataset`: Dataset name (mnli, snli)
- `--model_type`: Base model type (bert)
- `--method`: Training method (base, base_llm)
- `--llm`: LLM to use for plugin (llama3.1)
- `--job`: Operation to perform (preprocessing, training, testing)

## Experimental Setup

The experiments compare two approaches:
1. **Baseline (`base`)**: Standard SLM fine-tuning
2. **PiFi (`base_llm`)**: SLM fine-tuning with frozen LLM layer plugin

Both approaches are evaluated on the same downstream tasks to demonstrate the effectiveness of the PiFi methodology.

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{kim-etal-2025-plug,
    title = "Plug-in and Fine-tuning: Bridging the Gap between Small Language Models and Large Language Models",
    author = "Kim, Kyeonghyun  and
      Jang, Jinhee  and
      Choi, Juhwan  and
      Lee, Yoonji  and
      Jin, Kyohoon  and
      Kim, YoungBin",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.271/",
    doi = "10.18653/v1/2025.acl-long.271",
    pages = "5434--5452",
    ISBN = "979-8-89176-251-0",
    abstract = "Large language models (LLMs) are renowned for their extensive linguistic knowledge and strong generalization capabilities, but their high computational demands make them unsuitable for resource-constrained environments. In contrast, small language models (SLMs) are computationally efficient but often lack the broad generalization capacity of LLMs. To bridge this gap, we propose PiFi, a novel framework that combines the strengths of both LLMs and SLMs to achieve high performance while maintaining efficiency. PiFi integrates a single frozen layer from an LLM into a SLM and fine-tunes the combined model for specific tasks, boosting performance without a significant increase in computational cost. We show that PiFi delivers consistent performance improvements across a range of natural language processing tasks, including both natural language understanding and generation. Moreover, our findings demonstrate PiFi{'}s ability to effectively leverage LLM knowledge, enhancing generalization to unseen domains and facilitating the transfer of linguistic abilities."
}
```
