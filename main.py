import torch
import argparse
import os
import json
import logging
from datetime import datetime

from config import Config, ModelConfig
from data_handler import DataHandler
from models import get_model
from train import Trainer
from utils import set_seed, load_config
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train intent classification model with regularization")
    
    # Model arguments
    parser.add_argument("--model_type", type=str, default="bert", choices=["bert", "roberta"],
                        help="Type of model to use (bert or roberta)")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased",
                        help="Pretrained model name or path")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate for the model")
    
    # Data arguments
    parser.add_argument("--dataset_name", type=str, default="banking77",
                        help="Dataset name to use")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to the data directory")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training and evaluation")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading")
    
    # Training arguments
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=0,
                        help="Number of warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--early_stopping", type=int, default=3,
                        help="Early stopping patience")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of steps for gradient accumulation")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Maximum gradient norm")
    parser.add_argument("--fp16", action="store_true", 
                        help="Use mixed precision training")
    
    # Regularization arguments
    parser.add_argument("--use_ib", action="store_true",
                        help="Use Information Bottleneck regularization")
    parser.add_argument("--ib_lambda", type=float, default=1e-4,
                        help="IB regularization strength")
    
    parser.add_argument("--use_afr", action="store_true",
                        help="Use Anchored Feature Regularization")
    parser.add_argument("--afr_lambda", type=float, default=1e-3,
                        help="AFR regularization strength")
    parser.add_argument("--afr_type", type=str, default="global", choices=["global", "class", "instance"],
                        help="Type of anchors to use for AFR")
    parser.add_argument("--afr_projection_dim", type=int, default=None,
                        help="Projection dimension for AFR (None for no projection)")

    # Other arguments
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save outputs (default: auto-generated)")
    parser.add_argument("--config_file", type=str, default=None,
                        help="Path to config file (overrides command line args)")

    return parser.parse_args()

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()

    # Create configuration
    if args.config_file:
        # Load config from file
        config = load_config(args.config_file)
    else:
        # Create config from command line args
        config = Config()

        # Set model config
        config.model.model_type = args.model_type
        config.model.model_name = args.model_name
        config.model.dropout = args.dropout

        # Set data config
        config.data.dataset_name = args.dataset_name
        if args.data_dir:
            config.data.data_dir = args.data_dir
        config.data.max_length = args.max_length
        config.data.batch_size = args.batch_size
        config.data.num_workers = args.num_workers

        # Set training config
        config.training.learning_rate = args.learning_rate
        config.training.epochs = args.epochs
        config.training.warmup_steps = args.warmup_steps
        config.training.weight_decay = args.weight_decay
        config.training.early_stopping_patience = args.early_stopping
        config.training.gradient_accumulation_steps = args.gradient_accumulation_steps
        config.training.max_grad_norm = args.max_grad_norm

        # Set regularizer config
        config.regularizers.use_ib = args.use_ib
        config.regularizers.ib_lambda = args.ib_lambda
        config.regularizers.use_afr = args.use_afr
        config.regularizers.afr_lambda = args.afr_lambda
        config.regularizers.anchor_type = args.afr_type
        config.regularizers.afr_projection_dim = args.afr_projection_dim

        # Set seed and device
        config.seed = args.seed
        config.device = args.device
        config.fp16 = args.fp16

    # Set output directory
    if args.output_dir:
        config.output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        regularizers = []
        if config.regularizers.use_ib:
            regularizers.append(f"IB_{config.regularizers.ib_lambda}")
        if config.regularizers.use_afr:
            regularizers.append(f"AFR-{config.regularizers.anchor_type}_{config.regularizers.afr_lambda}")
        
        regularizer_str = "_".join(regularizers) if regularizers else "baseline"
        config.output_dir = f"/ssd_scratch/cvit/lalitha/InfoBert/outputs/{config.model.model_type}_{regularizer_str}_{timestamp}"

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Save configuration
    with open(os.path.join(config.output_dir, "config.json"), "w") as f:
        # Convert config to dict for saving
        config_dict = {
            "model": vars(config.model),
            "data": vars(config.data),
            "regularizers": vars(config.regularizers),
            "training": vars(config.training),
            "seed": config.seed,
            "output_dir": config.output_dir,
            "device": config.device,
            "fp16": config.fp16
        }
        json.dump(config_dict, f, indent=2)

    # Set up logging to file
    file_handler = logging.FileHandler(os.path.join(config.output_dir, "train.log"))
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    logger.addHandler(file_handler)

    # Log configuration
    logger.info(f"Configuration: {config_dict}")

    # Set seed for reproducibility
    set_seed(config.seed)

    # Load and prepare data
    logger.info("Loading and preparing data...")
    data_handler = DataHandler(config)
    data_handler.load_data()
    _,_,_ = data_handler.prepare_data()
    train_loader, val_loader, test_loader = data_handler.get_dataloaders()

    # # Get label information
    # num_labels = data_handler.get_num_labels()
    # intent_names = data_handler.get_intent_names()
    # logger.info(f"Number of intent labels: {num_labels}")

    # # Create model
    # logger.info(f"Creating {config.model.model_type} model: {config.model.model_name}")
    # model = get_model(config, num_labels)

    # # Move model to device
    # device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    # model.to(device)

    # # Log model info
    # total_params = sum(p.numel() for p in model.parameters())
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # logger.info(f"Model created with {total_params:,} parameters ({trainable_params:,} trainable)")

    # # Log regularization settings
    # if config.regularizers.use_ib:
    #     logger.info(f"Using Information Bottleneck with lambda={config.regularizers.ib_lambda}")
    # if config.regularizers.use_afr:
    #     logger.info(f"Using Anchored Feature Regularization with type={config.regularizers.anchor_type}, "
    #                f"lambda={config.regularizers.afr_lambda}, "
    #                f"projection_dim={config.regularizers.afr_projection_dim}")

    # # Create trainer and train
    # logger.info("Initializing trainer...")
    # trainer = Trainer(
    #     model=model,
    #     config=config,
    #     train_dataloader=train_loader,
    #     val_dataloader=val_loader,
    #     test_dataloader=test_loader,
    #     num_labels=num_labels,
    #     intent_names=intent_names
    # )

    # # Train and evaluate
    # logger.info("Starting training...")
    # results = trainer.train()

    # # Log final results
    # logger.info("***** Training completed *****")
    # logger.info(f"Best epoch: {results['best_epoch']}")
    # logger.info(f"Best validation accuracy: {results['best_accuracy']:.4f}")
    # logger.info(f"Test accuracy: {results['test_metrics']['accuracy']:.4f}")
    # logger.info(f"Test F1 (macro): {results['test_metrics']['f1_macro']:.4f}")

    # # Save results
    # with open(os.path.join(config.output_dir, "results.json"), "w") as f:
    #     json.dump({
    #         "best_epoch": results["best_epoch"],
    #         "best_accuracy": float(results["best_accuracy"]),
    #         "test_metrics": {k: float(v) for k, v in results["test_metrics"].items() if isinstance(v, (int, float))}
    #     }, f, indent=2)

    # logger.info(f"Results saved to {config.output_dir}")

if __name__ == "__main__":
    main()
