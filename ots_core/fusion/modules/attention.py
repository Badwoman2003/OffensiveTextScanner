"""Fused self-attention for ViLT encoder layers.

Combines:
- Horizontal fusion via ``FusedQKVLinear`` (single GEMM for Q/K/V).
- Vertical fusion via ``flash_attention`` (softmax + value aggregation in one kernel).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ots_core.fusion.kernels.fused_attention import flash_attention
from ots_core.fusion.kernels.fused_qkv import FusedQKVLinear


class FusedViltSelfAttention(nn.Module):
    """Drop-in replacement for HF ``ViltSelfAttention``.

    Expected to be installed by ``patch_vilt`` — it reads the three sibling ``nn.Linear``s from
    the host module and builds a ``FusedQKVLinear`` whose weight is the row-wise concatenation.
    """

    def __init__(self, num_heads: int, head_dim: int, qkv: FusedQKVLinear, attn_dropout: float = 0.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_size = num_heads * head_dim
        self.qkv = qkv
        self.attn_dropout = attn_dropout

    @classmethod
    def from_hf(cls, hf_attention) -> "FusedViltSelfAttention":
        qkv = FusedQKVLinear.from_three(hf_attention.query, hf_attention.key, hf_attention.value)
        return cls(
            num_heads=hf_attention.num_attention_heads,
            head_dim=hf_attention.attention_head_size,
            qkv=qkv,
            attn_dropout=hf_attention.dropout.p if hasattr(hf_attention, "dropout") else 0.0,
        )

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        return x.view(B, N, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, head_mask=None, output_attentions: bool = False):
        q, k, v = self.qkv(hidden_states)
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        if attention_mask is not None and attention_mask.dim() == 4:
            # HF passes (B, 1, 1, N) with -inf for mask; convert to (B, N) int mask
            attention_mask = (attention_mask[:, 0, 0, :] >= 0).to(torch.int32)
        out = flash_attention(q, k, v, attention_mask=attention_mask)
        out = out.transpose(1, 2).contiguous().view(hidden_states.shape[0], hidden_states.shape[1], self.hidden_size)

        if output_attentions:
            # Fused kernel does not materialise attention weights; return None like xFormers does.
            return out, None
        return (out,)
