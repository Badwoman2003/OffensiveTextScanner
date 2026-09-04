"""Vertical fusion: attention score + mask + softmax + dropout + value aggregation.

Flash-Attention-style tiled kernel. ViLT's combined text-patch sequence length is ~250 which
fits nicely into a single SM's shared memory when tiled at BLOCK=64.

For numerical stability we track ``m`` (running row max) and ``l`` (running denom) across tiles.
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


if HAS_TRITON:

    @triton.jit
    def _flash_attn_kernel(
        Q_ptr, K_ptr, V_ptr, OUT_ptr, MASK_ptr,
        BATCH, HEADS, SEQ, HEAD_DIM,
        sm_scale,
        stride_qb, stride_qh, stride_qn, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_on, stride_od,
        stride_mb, stride_mn,
        HAS_MASK: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        b = pid_bh // HEADS
        h = pid_bh % HEADS

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)

        q_ptrs = Q_ptr + b * stride_qb + h * stride_qh + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
        q = tl.load(q_ptrs, mask=(offs_m[:, None] < SEQ) & (offs_d[None, :] < HEAD_DIM), other=0.0)

        m_i = tl.full([BLOCK_M], -1e30, dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

        for n0 in range(0, SEQ, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            k_ptrs = K_ptr + b * stride_kb + h * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
            v_ptrs = V_ptr + b * stride_vb + h * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
            k = tl.load(k_ptrs, mask=(offs_n[:, None] < SEQ) & (offs_d[None, :] < HEAD_DIM), other=0.0)
            v = tl.load(v_ptrs, mask=(offs_n[:, None] < SEQ) & (offs_d[None, :] < HEAD_DIM), other=0.0)

            qk = tl.dot(q, tl.trans(k)) * sm_scale  # (BLOCK_M, BLOCK_N)
            if HAS_MASK:
                m_ptrs = MASK_ptr + b * stride_mb + offs_n * stride_mn
                m_vals = tl.load(m_ptrs, mask=offs_n < SEQ, other=0)
                qk = tl.where(m_vals[None, :] == 0, -1e30, qk)

            m_new = tl.maximum(m_i, tl.max(qk, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new

        acc = acc / l_i[:, None]
        o_ptrs = OUT_ptr + b * stride_ob + h * stride_oh + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od
        tl.store(o_ptrs, acc.to(OUT_ptr.dtype.element_ty),
                 mask=(offs_m[:, None] < SEQ) & (offs_d[None, :] < HEAD_DIM))


def flash_attention(q, k, v, attention_mask=None):
    """q,k,v: (B, H, N, D). attention_mask: (B, N) with 1=keep, 0=masked. Returns (B, H, N, D)."""
    B, H, N, D = q.shape
    if not (HAS_TRITON and q.is_cuda):
        scale = 1.0 / math.sqrt(D)
        scores = (q @ k.transpose(-1, -2)) * scale
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask[:, None, None, :] == 0, float("-inf"))
        attn = scores.softmax(dim=-1)
        return attn @ v

    out = torch.empty_like(q)
    scale = 1.0 / math.sqrt(D)
    mask_ptr = attention_mask if attention_mask is not None else q.new_zeros((B, N), dtype=torch.int32)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = triton.next_power_of_2(D)

    grid = (triton.cdiv(N, BLOCK_M), B * H)
    _flash_attn_kernel[grid](
        q, k, v, out, mask_ptr.to(torch.int32),
        B, H, N, D,
        scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        mask_ptr.stride(0), mask_ptr.stride(1),
        HAS_MASK=attention_mask is not None,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
    )
    return out


class FlashAttentionModule(nn.Module):
    """Drop-in replacement for HF ViLT ``ViltSelfAttention``.forward on GPU."""

    def __init__(self, num_heads: int, head_dim: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

    def forward(self, q, k, v, attention_mask=None):
        return flash_attention(q, k, v, attention_mask=attention_mask)
