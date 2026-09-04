"""Classification heads + auxiliary single-modality heads used by Gradient-Blending."""
from __future__ import annotations

import torch
import torch.nn as nn


class OffenseClassificationHead(nn.Module):
    """LayerNorm + Dropout + Linear.

    Single hidden-to-label projection kept deliberately small so the head can't "memorise"
    around a weak backbone.
    """

    def __init__(self, hidden_size: int, num_labels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.act = nn.GELU()
        self.out = nn.Linear(hidden_size, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.dropout(self.act(self.dense(x)))
        return self.out(x)


class ModalityHead(nn.Module):
    """Tiny head operating on a single modality's pooled representation, used by
    Gradient-Blending (Wang et al. CVPR'20) to monitor unimodal OGR."""

    def __init__(self, hidden_size: int, num_labels: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
