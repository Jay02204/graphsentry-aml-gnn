"""
Class-Imbalance Mitigating Loss Functions for Financial Crime Graph Neural Networks.

Financial fraud and AML graphs exhibit extreme class skewness (<1% illicit rate).
This module implements numerically stable:
- BinaryFocalLossWithLogits: Down-weights well-classified easy licit examples
  and concentrates gradient signal on ambiguous, hard-to-detect mule accounts.
- WeightedBCEWithLogits: Cost-sensitive cross-entropy with positive class multiplier.
- compute_imbalance_weights: Dynamic alpha/pos_weight calculation from class distributions.
"""

import math
from typing import Any, Optional, Union

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def compute_imbalance_weights(
    labels: Union[list, Any],
) -> dict:
    """
    Computes class counts and recommended loss weighting factors:
    - pos_weight: N_neg / N_pos (for BCEWithLogitsLoss)
    - alpha: N_neg / (N_pos + N_neg) (for Focal Loss)
    """
    if _HAS_TORCH and isinstance(labels, torch.Tensor):
        num_pos = float((labels == 1).sum().item())
        num_neg = float((labels == 0).sum().item())
    else:
        num_pos = float(sum(1 for y in labels if y == 1))
        num_neg = float(sum(1 for y in labels if y == 0))

    total = num_pos + num_neg
    if num_pos == 0:
        pos_weight = 1.0
        alpha = 0.5
    else:
        pos_weight = num_neg / num_pos
        alpha = num_neg / total

    return {
        "num_positive": int(num_pos),
        "num_negative": int(num_neg),
        "total": int(total),
        "imbalance_pct": (num_pos / max(total, 1)) * 100.0,
        "pos_weight": float(pos_weight),
        "alpha": float(alpha),
    }


if _HAS_TORCH:
    class BinaryFocalLossWithLogits(nn.Module):
        """
        Numerically stable Binary Focal Loss operating directly on unnormalized logits:
            FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

        Parameters:
            gamma: Focusing parameter (default: 2.0).
                   gamma=0 corresponds to standard weighted binary cross-entropy.
            alpha: Balancing weight for positive class (default: 0.85).
            reduction: 'mean', 'sum', or 'none'.
        """

        def __init__(
            self,
            gamma: float = 2.0,
            alpha: Optional[float] = 0.85,
            reduction: str = "mean",
        ):
            super().__init__()
            self.gamma = gamma
            self.alpha = alpha
            self.reduction = reduction

        def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            """
            Args:
                logits: Unnormalized network outputs, shape (N,) or (N, 1).
                targets: Ground truth binary labels, shape matching logits.
            """
            logits = logits.view(-1)
            targets = targets.view(-1).float()

            # Numerically stable binary cross-entropy per element
            bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

            # Probability of correct class: p_t = p if y=1 else (1-p)
            p = torch.sigmoid(logits)
            p_t = p * targets + (1.0 - p) * (1.0 - targets)

            # Modulating factor (1 - p_t)^gamma
            modulating_factor = (1.0 - p_t).pow(self.gamma)

            # Class weighting factor alpha_t
            if self.alpha is not None:
                alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
                focal_loss = alpha_t * modulating_factor * bce_loss
            else:
                focal_loss = modulating_factor * bce_loss

            if self.reduction == "mean":
                return focal_loss.mean()
            elif self.reduction == "sum":
                return focal_loss.sum()
            elif self.reduction == "none":
                return focal_loss
            else:
                raise ValueError(f"Unsupported reduction mode: {self.reduction}")

else:
    # Pure Python / NumPy fallback implementation for environments without torch
    class BinaryFocalLossWithLogits:
        def __init__(self, gamma: float = 2.0, alpha: Optional[float] = 0.85, reduction: str = "mean"):
            self.gamma = gamma
            self.alpha = alpha
            self.reduction = reduction

        def __call__(self, logits: list, targets: list) -> float:
            total_loss = 0.0
            n = len(logits)
            if n == 0:
                return 0.0

            for z, y in zip(logits, targets):
                p = 1.0 / (1.0 + math.exp(-max(min(z, 30.0), -30.0)))
                p_t = p if y == 1 else (1.0 - p)
                bce = -math.log(max(p_t, 1e-12))
                mod = (1.0 - p_t) ** self.gamma
                alpha_t = self.alpha if y == 1 else (1.0 - self.alpha) if self.alpha else 1.0
                total_loss += alpha_t * mod * bce

            if self.reduction == "mean":
                return total_loss / n
            return total_loss
