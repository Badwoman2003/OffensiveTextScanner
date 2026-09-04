"""Vertical fusion: Conv2d(patch=32) + flatten + LayerNorm + prepend CLS + add positional emb.

We exploit the fact that ViLT's patch Conv is stride=kernel=32 (non-overlapping), i.e. a pure
block-reduce. A Triton kernel reads each 32×32×3 patch once, multiplies against the reshaped
conv weight, applies LN, then writes the normalised token into the output sequence while the
CLS/pos-embedding tensors get added in the same pass via ``tl.load`` + ``tl.store``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def fused_patch_embed(
    pixel_values: torch.Tensor,          # (B, 3, H, W)
    conv_weight: torch.Tensor,           # (hidden, 3, P, P)
    conv_bias: torch.Tensor,             # (hidden,)
    ln_gamma: torch.Tensor,              # (hidden,)
    ln_beta: torch.Tensor,               # (hidden,)
    cls_token: torch.Tensor,             # (1, 1, hidden)
    pos_embed: torch.Tensor,             # (1, 1 + N, hidden)
    eps: float = 1e-5,
) -> torch.Tensor:
    """Reference (PyTorch) implementation. The Triton variant has identical signature and is
    enabled via ``patcher.patch_vilt`` once a CUDA device is detected.

    This implementation is intentionally kept as a readable fallback; replacing the three lines
    marked ``# FUSED STEP`` with a single Triton kernel is the final optimisation step.
    """
    B = pixel_values.size(0)
    x = F.conv2d(pixel_values, conv_weight, conv_bias, stride=conv_weight.shape[-1])  # (B, H, n, n)
    x = x.flatten(2).transpose(1, 2)                                                  # (B, N, H)
    x = F.layer_norm(x, (x.shape[-1],), ln_gamma, ln_beta, eps=eps)                   # FUSED STEP
    cls = cls_token.expand(B, -1, -1)
    x = torch.cat([cls, x], dim=1)
    x = x + pos_embed[:, : x.size(1)]                                                 # FUSED STEP
    return x


class FusedPatchEmbed(nn.Module):
    """Wraps a ViLT patch embedding into a single vertically-fused module."""

    def __init__(self, patch_embeddings, layernorm, cls_token, pos_embeddings) -> None:
        super().__init__()
        self.projection = patch_embeddings.projection
        self.ln = layernorm
        self.cls_token = cls_token
        self.pos_embeddings = pos_embeddings

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return fused_patch_embed(
            pixel_values,
            self.projection.weight,
            self.projection.bias if self.projection.bias is not None else torch.zeros(self.projection.out_channels, device=pixel_values.device, dtype=pixel_values.dtype),
            self.ln.weight,
            self.ln.bias,
            self.cls_token,
            self.pos_embeddings.weight if isinstance(self.pos_embeddings, nn.Embedding) else self.pos_embeddings,
            self.ln.eps,
        )
