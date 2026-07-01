"""
Custom loss functions for multi-attribute classification.

Provides:
  - FocalLoss:   down-weights easy examples, focuses on hard/misclassified ones.
  - GCELoss:     Generalized Cross Entropy — bounded loss robust to label noise.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., ICCV 2017).

    Down-weights well-classified examples (those with high confidence) so
    the model focuses on hard, misclassified examples.  Particularly useful
    for highly imbalanced datasets and noisy labels.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where p_t is the model's estimated probability for the target class.
    gamma >= 0 — gamma=0 is equivalent to CrossEntropyLoss.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits, targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction='none',
        )
        pt = (-ce).exp()
        focal_weight = (1 - pt) ** self.gamma
        loss = focal_weight * ce
        return loss.mean()


class GCELoss(nn.Module):
    """
    Generalized Cross Entropy Loss (Zhang & Sabuncu, NeurIPS 2018).

    GCE applies a Box-Cox transformation to the probability of the target
    class.  When q < 1, the loss is bounded and less sensitive to outliers
    than standard CE (which is unbounded).  q → 1 recovers standard CE,
    while q → 0 approaches Mean Absolute Error (MAE, bounded).

    GCE(p_t) = (1 - p_t^q) / q      for q > 0
    GCE(p_t) = -log(p_t)            for q = 0  (standard CE)

    Typical q values:
        q = 0.7    mild noise robustness
        q = 0.5    strong noise robustness
        q = 0.0    standard CrossEntropyLoss

    References:
        Zhang & Sabuncu, "Generalized Cross Entropy Loss for Training Deep
        Neural Networks with Noisy Labels", NeurIPS 2018.
    """

    def __init__(
        self,
        q: float = 0.7,
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.q = q
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.q == 0.0:
            return F.cross_entropy(
                logits, targets,
                weight=self.weight,
                label_smoothing=self.label_smoothing,
            )

        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        # Gather probability of the target class
        one_hot = F.one_hot(targets, num_classes=logits.size(-1))
        p_t = (probs * one_hot).sum(dim=-1)

        # GCE = (1 - p_t^q) / q
        loss = (1.0 - p_t ** self.q) / self.q

        # Apply class weights if provided
        if self.weight is not None:
            class_weights = (one_hot * self.weight.unsqueeze(0)).sum(dim=-1)
            loss = loss * class_weights

        return loss.mean()
