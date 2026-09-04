"""Horizontal fusion: fused QKV projection.

Instead of three separate ``nn.Linear(H, H)`` we concatenate weight/bias along the out-features
dim and execute **one** GEMM. A Triton kernel then writes Q / K / V to three contiguous tiles
of the output tensor, saving two memory round-trips.

Forward layout::

    X     : (B, N, H)           - input
    W_qkv : (3H, H)             - stacked weights [W_q; W_k; W_v]
    b_qkv : (3H,)               - stacked biases
    Output: (B, N, 3H) split view -> Q, K, V  each (B, N, H)
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover - CI without triton
    triton = None
    tl = None
    HAS_TRITON = False


# ------------------------------ Triton kernel ---------------------------------------------------

if HAS_TRITON:

    @triton.jit
    def _qkv_fwd_kernel(
        X_ptr, W_ptr, B_ptr, OUT_ptr,
        M, K, N3,
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        stride_om, stride_on,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        """One GEMM writing Q | K | V side-by-side along the N3 dimension."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k_idx = k0 + offs_k
            x = tl.load(X_ptr + offs_m[:, None] * stride_xm + k_idx[None, :] * stride_xk,
                        mask=(offs_m[:, None] < M) & (k_idx[None, :] < K), other=0.0)
            w = tl.load(W_ptr + offs_n[:, None] * stride_wn + k_idx[None, :] * stride_wk,
                        mask=(offs_n[:, None] < N3) & (k_idx[None, :] < K), other=0.0)
            acc += tl.dot(x, tl.trans(w))

        b = tl.load(B_ptr + offs_n, mask=offs_n < N3, other=0.0)
        acc = acc + b[None, :]

        tl.store(OUT_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
                 acc.to(OUT_ptr.dtype.element_ty),
                 mask=(offs_m[:, None] < M) & (offs_n[None, :] < N3))


def fused_qkv(x: torch.Tensor, w_qkv: torch.Tensor, b_qkv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """x: (B, N, H); w_qkv: (3H, H); b_qkv: (3H,) -> Q,K,V each (B, N, H)."""
    B, N, H = x.shape
    out_dim = w_qkv.shape[0]
    assert out_dim == 3 * H, f"w_qkv first dim must be 3H, got {out_dim}"
    M = B * N

    if not (HAS_TRITON and x.is_cuda):
        y = torch.nn.functional.linear(x, w_qkv, b_qkv)  # (B, N, 3H)
        q, k, v = y.split(H, dim=-1)
        return q.contiguous(), k.contiguous(), v.contiguous()

    x2 = x.reshape(M, H).contiguous()
    out = torch.empty((M, out_dim), device=x.device, dtype=x.dtype)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(out_dim, BLOCK_N))
    _qkv_fwd_kernel[grid](
        x2, w_qkv, b_qkv, out,
        M, H, out_dim,
        x2.stride(0), x2.stride(1),
        w_qkv.stride(0), w_qkv.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    y = out.view(B, N, out_dim)
    q, k, v = y.split(H, dim=-1)
    return q.contiguous(), k.contiguous(), v.contiguous()


class FusedQKVLinear(nn.Module):
    """Drop-in replacement for three parallel ``nn.Linear(H, H)``."""

    def __init__(self, hidden: int, bias: bool = True) -> None:
        super().__init__()
        self.hidden = hidden
        self.weight = nn.Parameter(torch.empty(3 * hidden, hidden))
        self.bias = nn.Parameter(torch.zeros(3 * hidden)) if bias else None
        nn.init.xavier_uniform_(self.weight)

    @classmethod
    def from_three(cls, q: nn.Linear, k: nn.Linear, v: nn.Linear) -> "FusedQKVLinear":
        assert q.in_features == k.in_features == v.in_features
        assert q.out_features == k.out_features == v.out_features == q.in_features
        module = cls(q.in_features, bias=q.bias is not None)
        with torch.no_grad():
            module.weight.copy_(torch.cat([q.weight, k.weight, v.weight], dim=0))
            if module.bias is not None:
                module.bias.copy_(torch.cat([q.bias, k.bias, v.bias], dim=0))
        return module

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return fused_qkv(x, self.weight, self.bias if self.bias is not None else torch.zeros(3 * self.hidden, device=x.device, dtype=x.dtype))


# Reference implementation used exclusively by numerical tests
def qkv_reference(x, wq, wk, wv, bq, bk, bv):
    return (
        torch.nn.functional.linear(x, wq, bq),
        torch.nn.functional.linear(x, wk, bk),
        torch.nn.functional.linear(x, wv, bv),
    )
