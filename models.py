import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from typing import Optional, Dict, Any, Tuple
from regularizers import InformationBottleneck, AnchoredFeatureRegularizer # Assuming this will be updated

# FeatureRegularizer class remains the same, so I'll omit it for brevity

class IntentClassifier(nn.Module):
    """Intent classification model with optional regularization"""

    def __init__(
        self,
        config: Dict[str, Any],
        num_labels: int
    ):
        super().__init__()

        # Load pre-trained model and configuration
        model_config = AutoConfig.from_pretrained(
            config.model.model_name,
            num_labels=num_labels
        )
        self.encoder = AutoModel.from_pretrained(
            config.model.model_name,
            config=model_config
        )

        self.hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)

        # Information Bottleneck
        self.use_ib = config.regularizers.use_ib
        if self.use_ib:
            # Define layers to project features to mu and logvar
            # The dimension of the latent space 'z' (bottleneck_dim) can be hidden_size or smaller.
            # For simplicity, let's assume the latent space z has the same dimension as hidden_size
            # so the classifier doesn't need to change its input dimension.
            # If you want a different bottleneck_dim, ensure the classifier can handle it.
            vib_bottleneck_dim = getattr(config.regularizers, 'vib_bottleneck_dim', self.hidden_size)

            self.fc_mu = nn.Linear(self.hidden_size, vib_bottleneck_dim)
            self.fc_logvar = nn.Linear(self.hidden_size, vib_bottleneck_dim)
            self.ib = InformationBottleneck( # Renamed ib_regularizer to ib for consistency
                lambda_value=config.regularizers.ib_lambda
            )
            # If vib_bottleneck_dim is different from self.hidden_size,
            # the classifier needs to take vib_bottleneck_dim as input.
            if vib_bottleneck_dim != self.hidden_size:
                self.classifier = nn.Linear(vib_bottleneck_dim, num_labels)
            else:
                self.classifier = nn.Linear(self.hidden_size, num_labels)
        else:
            self.classifier = nn.Linear(self.hidden_size, num_labels)


        # Anchored Feature Regularization
        self.use_afr = config.regularizers.use_afr
        if self.use_afr:
            # AFR regularizes the 'features' directly from the encoder.
            # If you wanted AFR to regularize 'z' from VIB, its input dim might need adjustment.
            afr_feature_dim = self.hidden_size # AFR operates on features before VIB bottleneck
            afr_projection_dim = config.regularizers.get('afr_projection_dim', None)

            self.afr = AnchoredFeatureRegularizer(
                num_classes=num_labels,
                feature_dim=afr_feature_dim,
                lambda_value=config.regularizers.afr_lambda,
                anchor_type=config.regularizers.anchor_type,
                projection_dim=afr_projection_dim # Pass the projection_dim here
            )
        print(f'inside intent classifier init ')
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        **kwargs
    ):
        # Encode inputs
        print(f'inside intentclassifier forward')
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids if hasattr(self.encoder.config, 'type_vocab_size') else None
        )

        # Get pooled output (CLS token)
        pooled_output = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs.last_hidden_state[:, 0]

        # Apply dropouts
        features = self.dropout(pooled_output) # These are the features before VIB transformation

        loss_dict = {}
        classification_features = features # Default: features passed to classifier

        if self.use_ib:
            mu = self.fc_mu(features)
            logvar = self.fc_logvar(features) # logvar is log(sigma^2)

            # Calculate IB loss using the regularizer instance
            # The InformationBottleneck forward method should compute the KL divergence
            ib_loss = self.ib(mu, logvar)
            loss_dict['ib_loss'] = ib_loss

            # Sample z using the reparameterization trick for the classifier
            # The InformationBottleneck reparameterize method should perform the sampling
            z_for_classifier = self.ib.reparameterize(mu, logvar)
            classification_features = z_for_classifier # Classifier will use 'z'
        
        if self.use_afr:
            # AFR typically regularizes the deterministic features from the encoder
            # If labels are needed for 'class' anchors, ensure they are present
            if labels is None and self.afr.anchor_type == 'class' and self.training:
                 raise ValueError("AFR with anchor_type 'class' requires labels during training.")
            
            # Only apply AFR if labels are available (for class anchors) or if not class anchor type
            if labels is not None or self.afr.anchor_type != 'class':
                 afr_loss = self.afr(features, labels if self.afr.anchor_type == 'class' else None) # Pass labels only if needed by AFR
                 loss_dict['afr_loss'] = afr_loss
            elif self.training: # If training and labels are None but required
                print("Warning: AFR with class anchors skipped due to missing labels during training.")


        # Classification
        logits = self.classifier(classification_features) # Uses 'z' if VIB active, else 'features'

        # Calculate classification loss if labels are provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            loss_dict['ce_loss'] = loss

            # Add regularization losses
            total_loss = loss
            for reg_loss_name, reg_loss_value in loss_dict.items():
                if reg_loss_name != 'ce_loss' and reg_loss_value is not None: # Check for None
                    total_loss += reg_loss_value

            loss_dict['total_loss'] = total_loss
            loss = total_loss

        return {
            'loss': loss,
            'logits': logits,
            'loss_dict': loss_dict,
            'hidden_states': outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            'attentions': outputs.attentions if hasattr(outputs, 'attentions') else None,
        }


def get_model(config, num_labels):
    """Factory function to create a model based on configuration"""
    print(f'inside models.py get model function')
    model = IntentClassifier(config, num_labels)
    return model