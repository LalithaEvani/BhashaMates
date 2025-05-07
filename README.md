# Enhancing Banking Intent Classification using Information Bottleneck (IB) and Anchored Feature Regularization (AFR)
This repository is created as part of the INLP project. 

This repository implements intent classification on the [Banking77](https://huggingface.co/datasets/banking77) dataset using large pre-trained models (BERT-large, RoBERTa-large) with optional regularization techniques: **Information Bottleneck (IB)** and **Anchored Feature Regularization (AFR)**.

---

## Project Structure

```
├── main.py                      # Main entry point to run training/evaluation
├── train.py                     # Trainer class handling the training loop
├── config.py                    # Configuration definitions (via Pydantic)
├── data_handler.py              # Dataset loading and preprocessing logic
├── models/
│   ├── __init__.py              # Returns model instances from configs
│   ├── bert_model.py            # BERT model with support for IB/AFR
│   └── roberta_large.py         # RoBERTa model with support for IB/AFR
├── regularizers.py
├── utils.py                     # Helper utilities (e.g., seed setting)
├── config_bert_large.yaml       # Config for BERT-large
├── config_roberta_large.yaml    # Config for RoBERTa-large
├── run.sh                       # Shell script for batch experiments
├── outputs/                     # Auto-generated outputs directory
├── logs/                        # Auto-generated logs directory
└── README.md                    # This file
```

---

## Running Experiments

### Logs and Outputs

- Logs for each run will be saved in the `logs/` directory.
- Model checkpoints, results, and configs are saved in the `outputs/` directory.

### 1. Using `main.py` with a configuration file

```bash
python main.py --config_file config_bert_large.yaml
# or
python main.py --config_file config_roberta_large.yaml
```

### 2. Overriding YAML configuration with CLI arguments

```bash
python main.py \
    --model_type bert \
    --model_name bert-base-uncased \
    --dataset_name banking77 \
    --epochs 5 \
    --batch_size 32 \
    --learning_rate 2e-5 \
    --use_ib \
    --ib_lambda 1e-4 \
    --use_afr \
    --afr_lambda 1e-3 \
    --afr_type class \
    --output_dir ./my_custom_output
```

### 3. Help Menu

To view all command-line arguments:

```bash
python main.py --help
```

---

## Key CLI Arguments

| Argument            | Description                                                  |
|---------------------|--------------------------------------------------------------|
| `--config_file`     | YAML config file path                                        |
| `--model_type`      | Model type (`bert` or `roberta`)                             |
| `--model_name`      | Pretrained model from HuggingFace (e.g. `bert-base-uncased`) |
| `--dataset_name`    | Dataset name (`banking77`)                                   |
| `--data_dir`        | Local data directory path                                    |
| `--output_dir`      | Output directory for logs/models                             |
| `--epochs`          | Number of training epochs                                    |
| `--batch_size`      | Batch size for training/evaluation                           |
| `--learning_rate`   | Learning rate for optimizer                                  |
| `--use_ib`          | Enable Information Bottleneck                                |
| `--ib_lambda`       | IB loss weight                                               |
| `--use_afr`         | Enable AFR                                                   |
| `--afr_lambda`      | AFR loss weight                                              |
| `--afr_type`        | AFR type (`class`, `global`, or `instance`)                  |
| `--afr_projection_dim` | AFR projection dimension                                |
| `--fp16`            | Enable mixed-precision training                              |
| `--seed`            | Random seed                                                  |
| `--device`          | `cuda` or `cpu`                                              |

## Outputs per Run

Each run (under `outputs/`) will include:

- `config.json`: Final config used (merged YAML and CLI).
- `train.log`: Full training log.
- `results.json`: Evaluation metrics and best epoch.
- `best_model/`: Best model checkpoint.

When using `run.sh`, top-level logs for each experiment go in the `logs/` directory.

---

## requirements.txt

```
torch
transformers
scikit-learn
numpy
tqdm
datasets
PyYAML
```

---

## Created By Bhashamates
- Anuska Maity
- Devansh Chaudhary
- Evani Lalitha 




