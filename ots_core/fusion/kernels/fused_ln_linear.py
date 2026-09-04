"""Vertical fusion: LayerNorm followed immediately by a Linear projection.

Saves the read/write of the normalised activations in HBM. The Triton kernel computes mean,
rsqrt, normalises on-chip, then multiplies by Wᵀ and adds bias in one pass.
"""
from __future__ import annotations

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


if HAS_TRITON:

    @triton.jit
    def _ln_linear_fwd_kernel(
        X_ptr, GAMMA_ptr, BETA_ptr, W_ptr, BLIN_ptr, OUT_ptr,
        M, H, N,
        eps,
        stride_xm, stride_xh,
        stride_wn, stride_wh,
        stride_om, stride_on,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_H: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        # 1) compute LN stats per row in BLOCK_M
        mean = tl.zeros((BLOCK_M,), dtype=tl.float32)
        m2 = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for h0 in range(0, H, BLOCK_H):
            offs_h = h0 + tl.arange(0, BLOCK_H)
            x = tl.load(X_ptr + offs_m[:, None] * stride_xm + offs_h[None, :] * stride_xh,
                        mask=(offs_m[:, None] < M) & (offs_h[None, :] < H), other=0.0).to(tl.float32)
            mean += tl.sum(x, axis=1)
        mean = mean / H

        for h0 in range(0, H, BLOCK_H):
            offs_h = h0 + tl.arange(0, BLOCK_H)
            x = tl.load(X_ptr + offs_m[:, None] * stride_xm + offs_h[None, :] * stride_xh,
                        mask=(offs_m[:, None] < M) & (offs_h[None, :] < H), other=0.0).to(tl.float32)
            diff = x - mean[:, None]
            m2 += tl.sum(diff * diff, axis=1)
        rstd = 1.0 / tl.sqrt(m2 / H + eps)

        # 2) fused normalise + matmul against W[out_n, H] + bias
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for h0 in range(0, H, BLOCK_H):
            offs_h = h0 + tl.arange(0, BLOCK_H)
            x = tl.load(X_ptr + offs_m[:, None] * stride_xm + offs_h[None, :] * stride_xh,
                        mask=(offs_m[:, None] < M) & (offs_h[None, :] < H), other=0.0).to(tl.float32)
            g = tl.load(GAMMA_ptr + offs_h, mask=offs_h < H, other=1.0).to(tl.float32)
            b = tl.load(BETA_ptr + offs_h, mask=offs_h < H, other=0.0).to(tl.float32)
            x_hat = (x - mean[:, None]) * rstd[:, None] * g[None, :] + b[None, :]

            w = tl.load(W_ptr + offs_n[:, None] * stride_wn + offs_h[None, :] * stride_wh,
                        mask=(offs_n[:, None] < N) & (offs_h[None, :] < H), other=0.0).to(tl.float32)
            acc += tl.dot(x_hat, tl.trans(w))

        bias_lin = tl.load(BLIN_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        acc = acc + bias_lin[None, :]

        tl.store(OUT_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
                 acc.to(OUT_ptr.dtype.element_ty),
                 mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def fused_ln_linear(x, gamma, beta, weight, bias, eps=1e-5):
    """x: (B, N, H); weight: (N_out, H); returns (B, N, N_out)."""
    if not (HAS_TRITON and x.is_cuda):
        x_n = torch.nn.functional.layer_norm(x, (x.shape[-1],), gamma, beta, eps=eps)
        return torch.nn.functional.linear(x_n, weight, bias)

    B, S, H = x.shape
    M = B * S
    N = weight.shape[0]
    x2 = x.reshape(M, H).contiguous()
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    BLOCK_M, BLOCK_N, BLOCK_H = 32, 64, 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _ln_linear_fwd_kernel[grid](
        x2, gamma, beta, weight, bias, out,
        M, H, N, eps,
        x2.stride(0), x2.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_H=BLOCK_H,
    )
    return out.view(B, S, N)


class FusedLayerNormLinear(nn.Module):
    def __init__(self, ln: nn.LayerNorm, lin: nn.Linear) -> None:
        super().__init__()
        assert ln.normalized_shape == (lin.in_features,), "LN/Linear dim mismatch"
        self.eps = ln.eps
        self.gamma = nn.Parameter(ln.weight.detach().clone())
        self.beta = nn.Parameter(ln.bias.detach().clone())
        self.weight = nn.Parameter(lin.weight.detach().clone())
        self.bias = nn.Parameter(
            lin.bias.detach().clone() if lin.bias is not None else torch.zeros(lin.out_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fused_ln_linear(x, self.gamma, self.beta, self.weight, self.bias, self.eps)
