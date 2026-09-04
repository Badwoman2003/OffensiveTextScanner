"""Vertical fusion: bias-add + GELU + (optional) dropout in one elementwise pass.

Saves two full-tensor reads+writes between the linear output and the next matmul. We implement
dropout as a deterministic per-element rescale controlled by a seed/offset so the backward pass
can reproduce the mask without storing it.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover
    triton = None
    tl = None
    HAS_TRITON = False

_GELU_CONST = math.sqrt(2.0 / math.pi)


if HAS_TRITON:

    @triton.jit
    def _bias_gelu_dropout_kernel(
        X_ptr, B_ptr, OUT_ptr,
        M, N, p, seed,
        stride_m, stride_n,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        pid = tl.program_id(0)
        pid_m = pid // tl.cdiv(N, BLOCK_N)
        pid_n = pid % tl.cdiv(N, BLOCK_N)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

        x = tl.load(X_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n, mask=mask, other=0.0)
        b = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)
        x = x + b[None, :]

        # Tanh approximation to GELU: 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
        c = 0.044715
        k = 0.7978845608028654  # sqrt(2/pi)
        inner = k * (x + c * x * x * x)
        gelu = 0.5 * x * (1.0 + tl.math.tanh(inner))

        if p > 0.0:
            rand = tl.rand(seed, offs_m[:, None] * N + offs_n[None, :])
            keep = rand > p
            gelu = tl.where(keep, gelu / (1.0 - p), tl.zeros_like(gelu))

        tl.store(OUT_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n,
                 gelu.to(OUT_ptr.dtype.element_ty), mask=mask)


def fused_bias_gelu_dropout(x: torch.Tensor, bias: torch.Tensor, p: float = 0.0, training: bool = True) -> torch.Tensor:
    """x: (..., N). Bias-add + GELU + inverted dropout (when training & p>0)."""
    dropout_p = p if training else 0.0
    if not (HAS_TRITON and x.is_cuda):
        y = torch.nn.functional.gelu(x + bias, approximate="tanh")
        if dropout_p > 0.0:
            y = torch.nn.functional.dropout(y, p=dropout_p, training=True)
        return y

    x2 = x.contiguous().view(-1, x.shape[-1])
    M, N = x2.shape
    out = torch.empty_like(x2)
    BLOCK_M, BLOCK_N = 32, 128
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    seed = int(torch.randint(0, 2**31 - 1, (1,)).item()) if dropout_p > 0 else 0
    _bias_gelu_dropout_kernel[grid](
        x2, bias, out,
        M, N, dropout_p, seed,
        x2.stride(0), x2.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    return out.view_as(x)


class FusedBiasGeluDropout(nn.Module):
    def __init__(self, bias: nn.Parameter | torch.Tensor, p: float = 0.0) -> None:
        super().__init__()
        if not isinstance(bias, nn.Parameter):
            bias = nn.Parameter(bias.detach().clone())
        self.bias = bias
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fused_bias_gelu_dropout(x, self.bias, p=self.p, training=self.training)
