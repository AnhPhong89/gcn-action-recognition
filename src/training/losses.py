"""
Loss functions for action recognition training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy with label smoothing to reduce overfitting.

    Args:
        smoothing: Label smoothing factor in [0, 1).
    """

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        assert 0.0 <= smoothing < 1.0
        self.smoothing = smoothing

    def forward(self, pred, target):
        """
        Args:
            pred: (N, C) logits.
            target: (N,) class indices.
        """
        num_classes = pred.size(-1)
        log_probs = F.log_softmax(pred, dim=-1)

        # Create smooth target distribution
        with torch.no_grad():
            smooth_target = torch.full_like(log_probs, self.smoothing / (num_classes - 1))
            smooth_target.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)

        loss = (-smooth_target * log_probs).sum(dim=-1).mean()
        return loss


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance.

    Reduces the loss contribution from easy examples and focuses
    on hard, misclassified examples.

    Args:
        gamma: Focusing parameter. Higher = more focus on hard examples.
        alpha: Class balancing weight (scalar or per-class tensor).
    """

    def __init__(self, gamma: float = 2.0, alpha: float = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, target):
        """
        Args:
            pred: (N, C) logits.
            target: (N,) class indices.
        """
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)  # probability of correct class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            focal_loss = self.alpha * focal_loss

        return focal_loss.mean()


def build_loss(loss_type: str = 'cross_entropy', **kwargs) -> nn.Module:
    """Factory function to create a loss module.

    Args:
        loss_type: One of 'cross_entropy', 'label_smoothing', 'focal'.
        **kwargs: Extra arguments for the loss constructor.

    Returns:
        nn.Module loss function.
    """
    if loss_type == 'cross_entropy':
        return nn.CrossEntropyLoss(**kwargs)
    elif loss_type == 'label_smoothing':
        smoothing = kwargs.get('smoothing', 0.1)
        return LabelSmoothingCrossEntropy(smoothing=smoothing)
    elif loss_type == 'focal':
        gamma = kwargs.get('gamma', 2.0)
        alpha = kwargs.get('alpha', None)
        return FocalLoss(gamma=gamma, alpha=alpha)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. "
                         f"Choose from: cross_entropy, label_smoothing, focal")
