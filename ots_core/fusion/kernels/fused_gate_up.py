"""Horizontal fusion for a GLU-style FFN first layer: ``gate = XW_g, up = XW_u``.

If the user converts ViLT's FFN to GEGLU / SwiGLU the two projections share the same input, so
we merge their weights along the output dim and split after GEMM. Unused by the stock ViLT
checkpoint but kept here because the plan explicitly lists it.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


def fused_gate_up(x: torch.Tensor, w_gate_up: torch.Tensor, b_gate_up: torch.Tensor | None) -> Tuple[torch.Tensor, torch.Tensor]:
    """x: (B, N, H); w_gate_up: (2*Hmid, H); returns gate, up each (B, N, Hmid)."""
    y = torch.nn.functional.linear(x, w_gate_up, b_gate_up)
    hmid = y.shape[-1] // 2
    gate, up = y.split(hmid, dim=-1)
    return gate.contiguous(), up.contiguous()


class FusedGateUpLinear(nn.Module):
    def __init__(self, hidden: int, intermediate: int, bias: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(2 * intermediate, hidden))
        self.bias = nn.Parameter(torch.zeros(2 * intermediate)) if bias else None
        nn.init.xavier_uniform_(self.weight)

    @classmethod
    def from_two(cls, gate: nn.Linear, up: nn.Linear) -> "FusedGateUpLinear":
        module = cls(gate.in_features, gate.out_features, bias=gate.bias is not None)
        with torch.no_grad():
            module.weight.copy_(torch.cat([gate.weight, up.weight], dim=0))
            if module.bias is not None:
                module.bias.copy_(torch.cat([gate.bias, up.bias], dim=0))
        return module

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return fused_gate_up(x, self.weight, self.bias)
