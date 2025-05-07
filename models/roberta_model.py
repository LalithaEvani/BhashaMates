"""
RoBERTa model with IB and AFR regularization for intent classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel
from typing import Dict, Tuple, Optional, List, Union
from regularizers import InformationBottleneck, AnchoredFeatureRegularizer

class RegularizedRobertaForIntentClassification(nn.Module):
    """
    RoBERTa model with Information Bottleneck and Anchored Feature Regularization
    for intent classification on Banking77 dataset
    """
    def __init__(self, model_name: str, num_labels: int, config):
        """
        Initialize the regularized RoBERTa model
        
        Args:
            model_name: Name or path of the pretrained RoBERTa model
            num_labels: Number of intent classes
            config: Configuration object with model and regularization settings
        """
        super(RegularizedRobertaForIntentClassification, self).__init__()
        
        # Load pretrained RoBERTa model
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(config.model.dropout)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_labels)
        
        # Configuration
        self.config = config
        
        # For Information Bottleneck
        if config.regularizers.use_ib:
            self.ib_mu = nn.Linear(self.roberta.config.hidden_size, self.roberta.config.hidden_size)
            self.ib_logvar = nn.Linear(self.roberta.config.hidden_size, self.roberta.config.hidden_size)
            self.ib_regularizer = InformationBottleneck(lambda_value=config.regularizers.ib_lambda)
        
        # For Anchored Feature Regularization
        if config.regularizers.use_afr:
            feature_dim = self.roberta.config.hidden_size
            projection_dim = config.regularizers.afr_projection_dim
            
            self.afr_regularizer = AnchoredFeatureRegularizer(
                num_classes=num_labels,
                feature_dim=feature_dim,
                lambda_value=config.regularizers.afr_lambda,
                anchor_type=config.regularizers.anchor_type,
                projection_dim=projection_dim
            )
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        labels: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None  # Included for API compatibility, RoBERTa doesn't use this
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with regularization
        
        Args:
            input_ids: Token ids
            attention_mask: Attention mask
            labels: Intent labels (optional)
            token_type_ids: Token type ids (ignored in RoBERTa)
            
        Returns:
            Dictionary with loss, logits, and other outputs
        """
        # Get RoBERTa outputs (ignoring token_type_ids as RoBERTa doesn't use them)
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get the sequence representation
        pooled_output = outputs.pooler_output
        if pooled_output is None:  # Some RoBERTa models don't have a pooler
            pooled_output = outputs.last_hidden_state[:, 0, :]  # Use the first token representation
        
        # Calculate Information Bottleneck if enabled
        if self.config.regularizers.use_ib:
            mu = self.ib_mu(pooled_output)
            logvar = self.ib_logvar(pooled_output)
            z = self.ib_regularizer.reparameterize(mu, logvar)
            features = z  # Use the IB-regularized representation
            ib_loss = self.ib_regularizer(mu, logvar)
        else:
            features = pooled_output
            ib_loss = 0.0
        
        # Apply dropout and get logits
        features = self.dropout(features)
        logits = self.classifier(features)
        
        # Calculate outputs
        outputs = {
            "logits": logits,
            "features": features,
        }
        
                    # Calculate loss if labels are provided
        if labels is not None:
            # Classification loss
            loss_fct = nn.CrossEntropyLoss()
            classification_loss = loss_fct(logits, labels)
            outputs["classification_loss"] = classification_loss
            
            # Total loss starts with classification loss
            total_loss = classification_loss
            
            # Add Information Bottleneck loss if enabled
            if self.config.regularizers.use_ib:
                outputs["ib_loss"] = ib_loss
                total_loss += ib_loss
            
            # Add Anchored Feature Regularization loss if enabled
            if self.config.regularizers.use_afr:
                # Update anchors first
                self.afr_regularizer.update_anchors(features.detach(), labels)
                
                # Calculate AFR loss
                afr_loss = self.afr_regularizer(features, labels)
                outputs["afr_loss"] = afr_loss
                total_loss += afr_loss
            
            outputs["loss"] = total_loss
        
        return outputs
    
    def predict(self, input_ids, attention_mask, token_type_ids=None):
        """
        Make predictions without calculating loss
        
        Args:
            input_ids: Token ids
            attention_mask: Attention mask
            token_type_ids: Token type ids (ignored in RoBERTa)
            
        Returns:
            Predictions (class indices)
        """
        outputs = self.forward(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        
        logits = outputs["logits"]
        return torch.argmax(logits, dim=1)