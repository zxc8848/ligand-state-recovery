"""
Multi-class Focal Loss for Stage 3 bond-type classification.

FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

Focal loss down-weights well-classified samples so that training focuses
on hard minority classes (double/triple bonds).
"""

import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Multi-class focal loss.

    Args:
        gamma:   Focusing parameter.  gamma=0 recovers standard cross-entropy.
        weight:  (C,) class weights (alpha).  If None, no per-class weighting.
        reduction: 'mean' | 'sum' | 'none'.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if weight is not None:
            self.register_buffer('weight', weight.float())
        else:
            self.weight = None

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        reduction: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Args:
            logits:    (N, C) unnormalized class scores.
            targets:   (N,)   integer class labels in [0, C).
            reduction: override the instance-level reduction if provided.

        Returns:
            Focal loss scalar (or per-sample tensor if reduction='none').
        """
        if logits.numel() == 0:
            return logits.sum() * 0.0

        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        ce = ce.clamp(max=100.0)   # prevent inf when model is over-confident on wrong class
        pt = torch.exp(-ce).clamp(max=1.0)   # clamp to [0,1]: float32 may give pt>1 when ce<0
        focal = (1.0 - pt) ** self.gamma * ce

        red = reduction if reduction is not None else self.reduction
        if red == 'mean':
            return focal.mean()
        if red == 'sum':
            return focal.sum()
        return focal

    @staticmethod
    def class_weights_from_counts(counts: torch.Tensor) -> torch.Tensor:
        """
        Compute tempered inverse-frequency class weights from per-class sample counts.

        Instead of raw inverse-frequency weighting, use a square-root tempering:

            weights[c] = sqrt(total / (num_classes * counts[c]))

        This still upweights rare classes, but avoids over-correcting and pushing
        common single bonds into higher-order classes.

        Args:
            counts: (C,) number of samples per class.

        Returns:
            (C,) normalized weight tensor.
        """
        total = counts.sum().float()
        num_classes = counts.shape[0]
        w = torch.sqrt(total / (num_classes * counts.float().clamp(min=1)))
        return w / w.mean()  # normalize so weights average to 1
