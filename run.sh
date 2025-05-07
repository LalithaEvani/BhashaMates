#!/bin/bash
# Script to run experiments with different models and regularization settings

# Make sure the script is executable (chmod +x run.sh)

# Create output directories
mkdir -p outputs logs

# 1. BERT-large-uncased with both IB and AFR
echo "Running BERT-large-uncased with IB and AFR regularization..."
python main.py --config_file config_bert_large.yaml > logs/bert_large_ib_afr.log 2>&1

# 2. BERT-large-uncased with only IB
echo "Running BERT-large-uncased with only IB regularization..."
python main.py --config_file config_bert_large.yaml --use_afr false > logs/bert_large_ib.log 2>&1

# 3. BERT-large-uncased with only AFR
echo "Running BERT-large-uncased with only AFR regularization..."
python main.py --config_file config_bert_large.yaml --use_ib false > logs/bert_large_afr.log 2>&1

# 4. BERT-large-uncased baseline (no regularization)
echo "Running BERT-large-uncased baseline (no regularization)..."
python main.py --config_file config_bert_large.yaml --use_ib false --use_afr false > logs/bert_large_baseline.log 2>&1

# 5. RoBERTa-large with both IB and AFR
echo "Running RoBERTa-large with IB and AFR regularization..."
python main.py --config_file config_roberta_large.yaml > logs/roberta_large_ib_afr.log 2>&1

# 6. RoBERTa-large with only IB
echo "Running RoBERTa-large with only IB regularization..."
python main.py --config_file config_roberta_large.yaml --use_afr false > logs/roberta_large_ib.log 2>&1

# 7. RoBERTa-large with only AFR
echo "Running RoBERTa-large with only AFR regularization..."
python main.py --config_file config_roberta_large.yaml --use_ib false > logs/roberta_large_afr.log 2>&1

# 8. RoBERTa-large baseline (no regularization)
echo "Running RoBERTa-large baseline (no regularization)..."
python main.py --config_file config_roberta_large.yaml --use_ib false --use_afr false > logs/roberta_large_baseline.log 2>&1

echo "All experiments completed! Check the logs directory for output logs."