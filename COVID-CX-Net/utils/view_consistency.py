"""
utils/view_consistency.py
=========================
View-Consistency Regularization (L_view) from
"Projection-Induced Domain Shift in Chest X-Ray Classification."

Equation 2 (main paper):
    L_view = (1/|P|) * Σ_{(x_AP, x_PA) ∈ P}
             ‖ h(x_AP)/‖h(x_AP)‖₂ − h(x_PA)/‖h(x_PA)‖₂ ‖₂²

where P is the set of matched AP/PA pairs and h(·) is the embedding
produced by the penultimate layer (before the classifier).

Total objective: L_total = L_CE + λ * L_view
Optimal λ selected by grid search on validation set; λ=0.1 found optimal
(Table 4, main paper).
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ---------------------------------------------------------------------------
# View-Consistency Loss
# ---------------------------------------------------------------------------

class ViewConsistencyLoss(nn.Module):
    """
    Normalised cosine embedding loss between paired AP/PA embeddings.

    Encourages the network to map AP and PA images of the same patient
    to nearby points on the unit hypersphere, thereby removing
    projection-specific features from the learned representation.

    Args:
        reduction: 'mean' (default) or 'sum'.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        assert reduction in ("mean", "sum"), \
            f"reduction must be 'mean' or 'sum', got '{reduction}'"
        self.reduction = reduction

    def forward(
        self,
        embed_ap: torch.Tensor,
        embed_pa: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            embed_ap: Embedding vectors for AP images, shape (B, D).
            embed_pa: Embedding vectors for PA images, shape (B, D).
                      Must be matched pairs (same patient, same batch position).

        Returns:
            Scalar loss tensor.
        """
        # L2-normalise onto unit hypersphere
        h_ap = F.normalize(embed_ap, p=2, dim=1)   # (B, D)
        h_pa = F.normalize(embed_pa, p=2, dim=1)   # (B, D)

        # Squared L2 distance between normalised embeddings
        # ‖h_AP − h_PA‖₂² = 2 − 2 cos(h_AP, h_PA)
        diff = h_ap - h_pa                           # (B, D)
        loss = (diff ** 2).sum(dim=1)                # (B,)

        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()


# ---------------------------------------------------------------------------
# Combined training objective
# ---------------------------------------------------------------------------

class ProjectionAwareLoss(nn.Module):
    """
    L_total = L_CE + λ * L_view

    Supports two training modes:
        1. Unpaired batch:  only L_CE is applied (embed_ap=None).
        2. Paired batch:    both L_CE and L_view are applied.

    Args:
        lambda_view: Weight λ for view-consistency term (default 0.1).
        num_classes: Number of output classes for CE loss.
        reduction:   Reduction mode for view-consistency loss.
    """

    def __init__(
        self,
        lambda_view: float = 0.1,
        num_classes: int = 3,
        reduction: str = "mean",
    ):
        super().__init__()
        self.lambda_view  = lambda_view
        self.ce_loss      = nn.CrossEntropyLoss()
        self.view_loss    = ViewConsistencyLoss(reduction=reduction)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        embed_ap: torch.Tensor | None = None,
        embed_pa: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            logits:    Model output logits, shape (B, num_classes).
            labels:    Ground-truth integer labels, shape (B,).
            embed_ap:  AP-image embeddings for view-consistency (B, D) or None.
            embed_pa:  PA-image embeddings for view-consistency (B, D) or None.

        Returns:
            (total_loss, loss_components_dict)
        """
        l_ce = self.ce_loss(logits, labels)
        components = {"ce": l_ce.item()}

        if embed_ap is not None and embed_pa is not None:
            l_view = self.view_loss(embed_ap, embed_pa)
            total  = l_ce + self.lambda_view * l_view
            components["view"] = l_view.item()
            components["lambda"] = self.lambda_view
        else:
            total = l_ce
            components["view"] = 0.0

        components["total"] = total.item()
        return total, components
