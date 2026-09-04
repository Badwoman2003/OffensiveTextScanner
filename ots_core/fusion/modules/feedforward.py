"""Fused FFN sublayers.

ViLT FFN = ``Linear(H, 4H) -> GELU -> Dropout -> Linear(4H, H) -> Dropout -> residual``.

We apply vertical fusion to the first sub-block (Linear + GELU + Dropout) using
``fused_bias_gelu_dropout``; the second Linear stays as-is since its output goes directly into
the residual add which the HF layer handles.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ots_core.fusion.kernels.fused_bias_gelu import fused_bias_gelu_dropout


class FusedViltIntermediate(nn.Module):
    """Replace ViLT's ``ViltIntermediate`` (Linear + GELU) with linear-without-bias + fused bias+GELU."""

    def __init__(self, dense: nn.Linear, dropout_p: float = 0.0) -> None:
        super().__init__()
        self.dense = nn.Linear(dense.in_features, dense.out_features, bias=False)
        with torch.no_grad():
            self.dense.weight.copy_(dense.weight)
        self.bias = nn.Parameter(
            dense.bias.detach().clone() if dense.bias is not None else torch.zeros(dense.out_features)
        )
        self.dropout_p = dropout_p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dense(x)
        return fused_bias_gelu_dropout(y, self.bias, p=self.dropout_p, training=self.training)


class FusedViltOutput(nn.Module):
    """ViLT output sub-block unchanged — placeholder so the patcher API is symmetric."""

    def __init__(self, dense: nn.Linear, dropout: nn.Dropout) -> None:
        super().__init__()
        self.dense = dense
        self.dropout = dropout

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.dense(hidden_states)) + input_tensor
