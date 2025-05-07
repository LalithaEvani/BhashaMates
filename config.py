"""
Configuration classes for the model and training process
"""
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ModelConfig:
    """Configuration for the model"""
    model_type: str = "bert"
    model_name: str = "bert-base-uncased"
    dropout: float = 0.1  # Added dropout parameter


@dataclass
class DataConfig:
    """Configuration for data loading and processing"""
    dataset_name: str = "banking77"
    data_dir: Optional[str] = None
    max_length: int = 128
    batch_size: int = 8
    num_workers: int = 8


@dataclass
class RegularizerConfig:
    """Configuration for regularization techniques"""
    # Information Bottleneck
    use_ib: bool = False
    ib_lambda: float = 1e-4
    
    # Anchored Feature Regularization
    use_afr: bool = False
    afr_lambda: float = 1e-3
    anchor_type: str = "global"  # 'global', 'class', or 'instance'
    afr_projection_dim: Optional[int] = None


@dataclass
class TrainingConfig:
    """Configuration for training process"""
    learning_rate: float = 2e-5
    epochs: int = 10
    warmup_steps: int = 0
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 3
    scheduler_type: str = "linear"
    evaluation_steps: int = 0  # 0 means evaluate after each epoch


@dataclass
class Config:
    """Main configuration class"""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    regularizers: RegularizerConfig = field(default_factory=RegularizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    seed: int = 42
    output_dir: str = "/scratch/anuska/outputs"
    log_level: str = "info"
    save_steps: int = 0  # 0 means save only the best model
    save_total_limit: int = 3  # Number of total saved checkpoints to keep
    device: str = "cuda"  # "cuda" or "cpu"
    fp16: bool = False  # Whether to use mixed precision training