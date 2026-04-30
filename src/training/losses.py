"""
Loss functions for Bird Call Detection
- FocalLoss: handles class imbalance better than CrossEntropy
- SoftLabelCrossEntropy: used with Mixup / CutMix soft labels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al. 2017) for multi-class classification.
    Down-weights easy examples so the model focuses on hard ones.

    gamma=0 reduces to standard cross-entropy.
    alpha (float or list) provides per-class weighting.
    """

    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None,
                 reduction: str = "mean", label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha          # [num_classes] tensor or None
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  [B, C] raw model output
            targets: [B]    integer class labels  OR  [B, C] soft labels
        """
        num_classes = logits.size(1)

        # Soft-label path (Mixup / CutMix)
        if targets.dim() == 2:
            log_probs = F.log_softmax(logits, dim=1)
            probs = torch.exp(log_probs)
            focal_weight = (1.0 - probs) ** self.gamma
            loss = -(focal_weight * log_probs * targets).sum(dim=1)
            if self.reduction == "mean":
                return loss.mean()
            elif self.reduction == "sum":
                return loss.sum()
            return loss

        # Hard-label path with optional label smoothing
        if self.label_smoothing > 0:
            smoothed = torch.full((logits.size(0), num_classes),
                                  self.label_smoothing / (num_classes - 1),
                                  device=logits.device)
            smoothed.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            log_probs = F.log_softmax(logits, dim=1)
            probs = torch.exp(log_probs)
            focal_weight = (1.0 - probs.gather(1, targets.unsqueeze(1))).squeeze(1) ** self.gamma
            loss = -(focal_weight * (log_probs * smoothed).sum(dim=1))
        else:
            log_probs = F.log_softmax(logits, dim=1)
            probs = torch.exp(log_probs)
            pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            focal_weight = (1.0 - pt) ** self.gamma
            ce = F.nll_loss(log_probs, targets, weight=self.alpha, reduction="none")
            loss = focal_weight * ce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class SoftLabelCrossEntropy(nn.Module):
    """Standard cross-entropy that accepts soft (one-hot blend) labels."""

    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.dim() == 1:
            return F.cross_entropy(logits, targets, label_smoothing=self.label_smoothing)
        log_probs = F.log_softmax(logits, dim=1)
        return -(targets * log_probs).sum(dim=1).mean()


def build_loss(config: dict, class_weights: Optional[torch.Tensor] = None) -> nn.Module:
    """Factory: returns the configured loss function."""
    tc = config["training"]
    ls = tc.get("label_smoothing", 0.0)
    if tc.get("use_focal_loss", True):
        alpha = class_weights
        return FocalLoss(
            gamma=tc.get("focal_loss_gamma", 2.0),
            alpha=alpha,
            label_smoothing=ls,
        )
    return SoftLabelCrossEntropy(label_smoothing=ls)
