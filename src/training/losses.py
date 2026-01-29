# Smart Industrial Defect Detection - Custom Loss Functions
"""
Custom loss functions for defect detection training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Reduces loss for well-classified samples.
    """
    
    def __init__(self, 
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 reduction: str = 'mean'):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for positive class
            gamma: Focusing parameter
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, 
                inputs: torch.Tensor, 
                targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Predicted logits (B, C)
            targets: Ground truth labels (B,)
        
        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks.
    Can also be used for classification with proper adaptation.
    """
    
    def __init__(self, smooth: float = 1.0):
        """
        Initialize Dice Loss.
        
        Args:
            smooth: Smoothing factor to avoid division by zero
        """
        super().__init__()
        self.smooth = smooth
    
    def forward(self, 
                inputs: torch.Tensor, 
                targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice loss.
        
        Args:
            inputs: Predicted probabilities (B, C, H, W) or (B, C)
            targets: Ground truth (B, H, W) or (B,)
        
        Returns:
            Dice loss value
        """
        # Convert to one-hot if needed
        if len(inputs.shape) == 2:  # Classification case
            probs = F.softmax(inputs, dim=1)
            targets_onehot = F.one_hot(targets, num_classes=inputs.size(1)).float()
        else:  # Segmentation case
            probs = torch.sigmoid(inputs)
            targets_onehot = targets.unsqueeze(1).float()
        
        # Flatten
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets_onehot.view(targets_onehot.size(0), -1)
        
        # Compute Dice coefficient
        intersection = (probs_flat * targets_flat).sum(dim=1)
        dice = (2. * intersection + self.smooth) / (
            probs_flat.sum(dim=1) + targets_flat.sum(dim=1) + self.smooth
        )
        
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    """
    Combination of multiple loss functions.
    """
    
    def __init__(self,
                 focal_weight: float = 1.0,
                 dice_weight: float = 0.0,
                 ce_weight: float = 0.0,
                 class_weights: Optional[torch.Tensor] = None):
        """
        Initialize combined loss.
        
        Args:
            focal_weight: Weight for focal loss
            dice_weight: Weight for dice loss
            ce_weight: Weight for cross-entropy loss
            class_weights: Class weights for CE loss
        """
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        
        self.focal_loss = FocalLoss()
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    
    def forward(self, 
                inputs: torch.Tensor, 
                targets: torch.Tensor) -> torch.Tensor:
        """Compute combined loss."""
        total_loss = 0.0
        
        if self.focal_weight > 0:
            total_loss += self.focal_weight * self.focal_loss(inputs, targets)
        
        if self.dice_weight > 0:
            total_loss += self.dice_weight * self.dice_loss(inputs, targets)
        
        if self.ce_weight > 0:
            total_loss += self.ce_weight * self.ce_loss(inputs, targets)
        
        return total_loss


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing loss for improved generalization.
    """
    
    def __init__(self, 
                 num_classes: int,
                 smoothing: float = 0.1):
        """
        Initialize label smoothing loss.
        
        Args:
            num_classes: Number of classes
            smoothing: Smoothing factor (0-1)
        """
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, 
                inputs: torch.Tensor, 
                targets: torch.Tensor) -> torch.Tensor:
        """Compute label smoothing loss."""
        log_probs = F.log_softmax(inputs, dim=-1)
        
        # Create smoothed labels
        targets_onehot = F.one_hot(targets, self.num_classes).float()
        targets_smooth = targets_onehot * self.confidence + \
                        self.smoothing / self.num_classes
        
        loss = -torch.sum(targets_smooth * log_probs, dim=-1)
        return loss.mean()


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for learning discriminative features.
    Useful for anomaly detection training.
    """
    
    def __init__(self, margin: float = 1.0):
        """
        Initialize contrastive loss.
        
        Args:
            margin: Margin for dissimilar pairs
        """
        super().__init__()
        self.margin = margin
    
    def forward(self,
                anchor: torch.Tensor,
                positive: torch.Tensor,
                negative: torch.Tensor) -> torch.Tensor:
        """
        Compute triplet contrastive loss.
        
        Args:
            anchor: Anchor features
            positive: Positive (similar) features
            negative: Negative (dissimilar) features
        
        Returns:
            Triplet loss value
        """
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()
