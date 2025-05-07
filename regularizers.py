import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class InformationBottleneck(nn.Module):
    def __init__(self, lambda_value):
        super(InformationBottleneck, self).__init__()
        self.lambda_value = float(lambda_value)

    def forward(self, mu, logvar):
        """
        Compute the KL divergence loss for the bottleneck.
        """
        # KL divergence: 0.5 * sum(mu^2 + exp(logvar) - 1 - logvar)
        # The formula used is equivalent: -0.5 * sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl_loss = kl_loss.mean() # Average across the batch
        return self.lambda_value * kl_loss

    def reparameterize(self, mu, logvar):
        """
        Sample z using the reparameterization trick.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class AnchoredFeatureRegularizer(nn.Module): # Inherit from nn.Module
    """
    Anchored Feature Regularizer (AFR)

    Encourages representations to be close to class anchors.
    Anchors are typically updated via Exponential Moving Average (EMA)
    by explicitly calling the `update_anchors` method during training.
    """
    def __init__(
        self,
        num_classes: int,
        feature_dim: int, # This is the input feature dimension to AFR
        lambda_value: float = 1e-3,
        anchor_type: str = "global", # 'global' or 'class'
        projection_dim: Optional[int] = None
    ):
        super().__init__() # Call superclass __init__

        self.num_classes = num_classes
        self.input_feature_dim = feature_dim # Dimension of features fed into AFR
        self.lambda_value = float(lambda_value)
        self.anchor_type = anchor_type
        if self.anchor_type not in ["global", "class", "instance"]:
            raise ValueError(f"Unsupported anchor_type: {anchor_type}")
        if self.anchor_type == "instance":
            print("Warning: AFR anchor_type 'instance' will behave like 'class'.")
            self.anchor_type = "class" # Treat instance as class for simplicity here

        self.projection_dim = projection_dim
        if projection_dim is not None:
            self.projection = nn.Linear(feature_dim, projection_dim)
            self.current_feature_dim = projection_dim # Dimension after projection
        else:
            self.projection = None
            self.current_feature_dim = feature_dim # Dimension of anchors

        # Register anchors and counts as buffers
        # They will be moved to the correct device with the model
        # and saved in the state_dict.
        if self.anchor_type == "class":
            self.register_buffer("class_anchors", torch.zeros(num_classes, self.current_feature_dim))
            self.register_buffer("class_counts", torch.zeros(num_classes, dtype=torch.long))
        
        # Global anchor is relevant for both 'global' and as a component for 'class' updates if desired
        # For simplicity, we'll only use it explicitly for anchor_type 'global'.
        # Initialize it even if anchor_type is 'class' for the update_anchors logic.
        self.register_buffer("global_anchor", torch.zeros(1, self.current_feature_dim))
        self.register_buffer("global_count", torch.tensor(0, dtype=torch.long)) # For EMA of global anchor


    def update_anchors(self, features: torch.Tensor, labels: Optional[torch.Tensor] = None, momentum: float = 0.9):
        """
        Update class and/or global anchors using EMA based on the current batch.
        This method should be called explicitly during training.

        Args:
            features: (batch_size, input_feature_dim) - Raw features before projection
            labels: (batch_size,) - Required if anchor_type is 'class'
            momentum: Momentum for updating anchor values
        """
        if self.projection is not None:
            features = self.projection(features)
        
        # Ensure features are on the same device as anchors (buffers should handle this, but good practice)
        # features = features.to(self.global_anchor.device) # Not strictly needed if model is on one device

        detached_features = features.detach()

        # Global anchor update
        batch_mean_global = detached_features.mean(dim=0, keepdim=True)
        if self.global_count == 0:
             self.global_anchor = batch_mean_global
        else:
            self.global_anchor = momentum * self.global_anchor + (1 - momentum) * batch_mean_global
        self.global_count += detached_features.size(0)


        if self.anchor_type == "class":
            if labels is None:
                raise ValueError("Labels are required to update class anchors.")
            
            # Ensure labels are on the correct device
            # labels = labels.to(self.class_anchors.device) # As above, less critical with buffers

            for c in range(self.num_classes):
                class_mask = (labels == c)
                if class_mask.sum() > 0:
                    class_features = detached_features[class_mask]
                    class_mean = class_features.mean(dim=0)
                    if self.class_counts[c] == 0:
                        self.class_anchors[c] = class_mean
                    else:
                        self.class_anchors[c] = momentum * self.class_anchors[c] + (1 - momentum) * class_mean
                    self.class_counts[c] += class_mask.sum()


    def get_target_anchors(self, batch_size: int, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get the appropriate anchors for the current batch based on `anchor_type`.

        Args:
            batch_size: Number of samples in the current batch.
            labels: Class labels (batch_size,). Required if anchor_type is 'class'.

        Returns:
            Anchor tensor for each sample (batch_size, current_feature_dim)
        """
        if self.anchor_type == "global":
            # global_anchor is (1, current_feature_dim), expand to batch size
            return self.global_anchor.expand(batch_size, -1)

        elif self.anchor_type == "class":
            if labels is None:
                raise ValueError("Labels are required to get class anchors.")
            # labels = labels.to(self.class_anchors.device) # Ensure device match
            return self.class_anchors[labels]
        else:
            # Should not happen due to init check
            raise ValueError(f"Internal error: Unknown anchor type: {self.anchor_type}")

    def forward(self, features: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the AFR regularization loss.

        Args:
            features: Feature tensor (batch_size, input_feature_dim) - Raw features before projection
            labels: Class labels (batch_size,). Required if anchor_type is 'class'.
                    Can be None if anchor_type is 'global'.

        Returns:
            Scaled AFR loss (a scalar tensor)
        """
        if features.size(0) == 0: # Handle empty batch
            return torch.tensor(0.0, device=features.device, requires_grad=True)
            
        if self.projection is not None:
            projected_features = self.projection(features)
        else:
            projected_features = features
        
        # Ensure labels are on the same device if provided
        if labels is not None and labels.device != projected_features.device:
            labels = labels.to(projected_features.device)

        target_anchors = self.get_target_anchors(projected_features.size(0), labels)
        
        # Ensure target_anchors are on the same device as projected_features
        # (Buffers should handle this, but an explicit check or .to() can be a safeguard)
        if target_anchors.device != projected_features.device:
            target_anchors = target_anchors.to(projected_features.device)

        # Compute L2 distance squared
        distances_sq = torch.sum((projected_features - target_anchors) ** 2, dim=1)
        afr_loss = distances_sq.mean()

        return self.lambda_value * afr_loss