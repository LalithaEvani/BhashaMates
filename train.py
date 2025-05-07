import os
import torch
import torch.nn as nn
import numpy as np
import logging
import time
from tqdm import tqdm
from typing import Dict, List, Tuple, Any, Optional
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

logger = logging.getLogger(__name__)

class Trainer:
    """Trainer class to handle training and evaluation"""
    
    def __init__(
        self,
        model,
        config,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        num_labels,
        intent_names
    ):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.num_labels = num_labels
        self.intent_names = intent_names
        
        # Set device
        if torch.cuda.is_available():
            self.device = torch.device(self.config.device)
            self.model = torch.nn.DataParallel(self.model)  # For multi-GPU setup
        else:
            self.device = torch.device("cpu")
        
        self.model.to(self.device)
        
        # Create optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Initialize tracking variables
        self.best_accuracy = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
    
    def _create_optimizer(self):
        """Create optimizer for training"""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.training.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        return AdamW(
            optimizer_grouped_parameters,
            lr=float(self.config.training.learning_rate),
            eps=1e-8
        )
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        num_training_steps = len(self.train_dataloader) * self.config.training.epochs
        num_warmup_steps = self.config.training.warmup_steps
        print(f'num of training steps {num_training_steps}')
        print(f'num of warmup steps {num_warmup_steps}')
        
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
            )
        
        return LambdaLR(self.optimizer, lr_lambda)
    
    def train(self):
        """Train the model"""
        logger.info("***** Starting training *****")
        logger.info(f"  Num examples = {len(self.train_dataloader.dataset)}")
        logger.info(f"  Num epochs = {self.config.training.epochs}")
        logger.info(f"  Batch size = {self.config.data.batch_size}")
        logger.info(f"  Using IB = {self.config.regularizers.use_ib}")
        logger.info(f"  Using AFR = {self.config.regularizers.use_afr}")
        
        for epoch in range(self.config.training.epochs):
            logger.info(f"Epoch {epoch+1}/{self.config.training.epochs}")
            
            # Training
            train_loss = self._train_epoch()
            logger.info(f"  Train loss: {train_loss:.4f}")
            
            # Validation
            val_metrics = self._evaluate(self.val_dataloader)
            logger.info(f"  Validation accuracy: {val_metrics['accuracy']:.4f}")
            # logger.info(f"  Validation F1 (macro): {val_metrics['f1_macro']:.4f}")
            
            # Check for improvement
            if val_metrics["accuracy"] > self.best_accuracy:
                self.best_accuracy = val_metrics["accuracy"]
                self.best_epoch = epoch + 1
                self.patience_counter = 0
                
                # Save best model
                self._save_model()
                logger.info(f"  New best model saved! Accuracy: {self.best_accuracy:.4f}")
            else:
                self.patience_counter += 1
                logger.info(f"  No improvement. Patience: {self.patience_counter}/{self.config.training.early_stopping_patience}")
                
                # Early stopping
                if self.patience_counter >= self.config.training.early_stopping_patience:
                    logger.info("Early stopping triggered!")
                    break
        
        # Load best model for testing
        self._load_model()
        
        # Test evaluation
        test_metrics = self._evaluate(self.test_dataloader)
        logger.info("***** Test Results *****")
        logger.info(f"  Test accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"  Test F1 (macro): {test_metrics['f1_macro']:.4f}")
        
        # Return results
        return {
            "best_epoch": self.best_epoch,
            "best_accuracy": self.best_accuracy,
            "test_metrics": test_metrics
        }
    
    def _train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        # Progress bar
        progress_bar = tqdm(self.train_dataloader, desc="Training")
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs["loss"]
            
            # Backward pass
            loss.backward()
            
            # Log loss components
            loss_dict = outputs.get("loss_dict", {})
            progress_desc = f"Loss: {loss.item():.4f}"
            if "ce_loss" in loss_dict:
                progress_desc += f" | CE: {loss_dict['ce_loss'].item():.4f}"
            if "ib_loss" in loss_dict:
                progress_desc += f" | IB: {loss_dict['ib_loss'].item():.4f}"
            if "afr_loss" in loss_dict:
                progress_desc += f" | AFR: {loss_dict['afr_loss'].item():.4f}"
            progress_bar.set_description(progress_desc)
            
            # Update parameters
            if (step + 1) % self.config.training.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item()
        if (step + 1) % self.config.training.gradient_accumulation_steps != 0:
            # Final gradient update for remaining steps
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
        # Calculate average loss
        avg_loss = total_loss / len(self.train_dataloader)
        return avg_loss
    
    def _evaluate(self, dataloader):
        """Evaluate the model on the given dataloader"""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                logits = outputs["logits"]
                
                # Get predictions
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                labels = batch["labels"].cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        f1_macro = f1_score(all_labels, all_preds, average="macro")
        f1_weighted = f1_score(all_labels, all_preds, average="weighted")
        precision = precision_score(all_labels, all_preds, average="macro")
        recall = recall_score(all_labels, all_preds, average="macro")
        
        # Create confusion matrix
        conf_matrix = confusion_matrix(all_labels, all_preds)
        
        # Return metrics
        return {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "precision": precision,
            "recall": recall,
            "confusion_matrix": conf_matrix
        }
    
    def _save_model(self):
        """Save the model"""
        # Create model directory
        model_dir = os.path.join(self.config.output_dir, "best_model")
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model and config
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(model_dir)
        else:
            torch.save(self.model.state_dict(), os.path.join(model_dir, "pytorch_model.bin"))
    
    def _load_model(self):
        """Load the best model"""
        model_dir = os.path.join(self.config.output_dir, "best_model")
        
        # Load model weights
        if hasattr(self.model, "from_pretrained"):
            self.model = type(self.model).from_pretrained(model_dir)
            self.model.to(self.device)
        else:
            self.model.load_state_dict(torch.load(os.path.join(model_dir, "pytorch_model.bin")))
