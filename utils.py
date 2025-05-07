"""
Utility functions for the project
"""
import os
import json
import random
import logging
import numpy as np
import torch
import yaml
from typing import Dict, Any, Optional
from config import Config, ModelConfig, DataConfig, RegularizerConfig, TrainingConfig


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(log_level: str = "info", log_file: Optional[str] = None):
    """Setup logging configuration"""
    # Map string log level to numeric value
    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL
    }
    level = level_map.get(log_level.lower(), logging.INFO)
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            *([] if log_file is None else [logging.FileHandler(log_file)])
        ]
    )


def load_config(config_path: str) -> Config:
    """Load configuration from file"""
    # Determine file type
    ext = os.path.splitext(config_path)[1].lower()
    
    if ext == '.json':
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
    elif ext in ['.yaml', '.yml']:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported config file extension: {ext}")
    
    # Create config objects
    config = Config()
    
    # Set model config
    if 'model' in config_dict:
        model_config = config_dict['model']
        config.model = ModelConfig(
            model_type=model_config.get('model_type', config.model.model_type),
            model_name=model_config.get('model_name', config.model.model_name)
        )
    
    # Set data config
    if 'data' in config_dict:
        data_config = config_dict['data']
        config.data = DataConfig(
            dataset_name=data_config.get('dataset_name', config.data.dataset_name),
            data_dir=data_config.get('data_dir', config.data.data_dir),
            max_length=data_config.get('max_length', config.data.max_length),
            batch_size=data_config.get('batch_size', config.data.batch_size),
            num_workers=data_config.get('num_workers', config.data.num_workers)
        )
    
    # Set regularizer config
    if 'regularizers' in config_dict:
        reg_config = config_dict['regularizers']
        config.regularizers = RegularizerConfig(
            use_ib=reg_config.get('use_ib', config.regularizers.use_ib),
            ib_lambda=reg_config.get('ib_lambda', config.regularizers.ib_lambda),
            use_afr=reg_config.get('use_afr', config.regularizers.use_afr),
            afr_lambda=reg_config.get('afr_lambda', config.regularizers.afr_lambda),
            anchor_type=reg_config.get('anchor_type', config.regularizers.anchor_type),
            afr_projection_dim=reg_config.get('afr_projection_dim', config.regularizers.afr_projection_dim)
        )
    
    # Set training config
    if 'training' in config_dict:
        train_config = config_dict['training']
        config.training = TrainingConfig(
            learning_rate=train_config.get('learning_rate', config.training.learning_rate),
            epochs=train_config.get('epochs', config.training.epochs),
            warmup_steps=train_config.get('warmup_steps', config.training.warmup_steps),
            weight_decay=train_config.get('weight_decay', config.training.weight_decay),
            gradient_accumulation_steps=train_config.get('gradient_accumulation_steps', 
                                                       config.training.gradient_accumulation_steps),
            max_grad_norm=train_config.get('max_grad_norm', config.training.max_grad_norm),
            early_stopping_patience=train_config.get('early_stopping_patience', 
                                                  config.training.early_stopping_patience),
            scheduler_type=train_config.get('scheduler_type', config.training.scheduler_type),
            evaluation_steps=train_config.get('evaluation_steps', config.training.evaluation_steps)
        )
    
    # Set other config parameters
    config.seed = config_dict.get('seed', config.seed)
    config.output_dir = config_dict.get('output_dir', config.output_dir)
    config.log_level = config_dict.get('log_level', config.log_level)
    config.save_steps = config_dict.get('save_steps', config.save_steps)
    config.save_total_limit = config_dict.get('save_total_limit', config.save_total_limit)
    config.device = config_dict.get('device', config.device)
    config.fp16 = config_dict.get('fp16', config.fp16)
    
    return config


def save_config(config: Config, output_path: str):
    """Save configuration to file"""
    # Convert config to dictionary
    config_dict = {
        'model': {
            'model_type': config.model.model_type,
            'model_name': config.model.model_name
        },
        'data': {
            'dataset_name': config.data.dataset_name,
            'data_dir': config.data.data_dir,
            'max_length': config.data.max_length,
            'batch_size': config.data.batch_size,
            'num_workers': config.data.num_workers
        },
        'regularizers': {
            'use_ib': config.regularizers.use_ib,
            'ib_lambda': config.regularizers.ib_lambda,
            'use_afr': config.regularizers.use_afr,
            'afr_lambda': config.regularizers.afr_lambda,
            'anchor_type': config.regularizers.anchor_type,
            'afr_projection_dim': config.regularizers.afr_projection_dim
        },
        'training': {
            'learning_rate': config.training.learning_rate,
            'epochs': config.training.epochs,
            'warmup_steps': config.training.warmup_steps,
            'weight_decay': config.training.weight_decay,
            'gradient_accumulation_steps': config.training.gradient_accumulation_steps,
            'max_grad_norm': config.training.max_grad_norm,
            'early_stopping_patience': config.training.early_stopping_patience,
            'scheduler_type': config.training.scheduler_type,
            'evaluation_steps': config.training.evaluation_steps
        },
        'seed': config.seed,
        'output_dir': config.output_dir,
        'log_level': config.log_level,
        'save_steps': config.save_steps,
        'save_total_limit': config.save_total_limit,
        'device': config.device,
        'fp16': config.fp16
    }
    
    # Determine file type from extension
    ext = os.path.splitext(output_path)[1].lower()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    # Save to file
    if ext == '.json':
        with open(output_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    elif ext in ['.yaml', '.yml']:
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    else:
        raise ValueError(f"Unsupported config file extension: {ext}")