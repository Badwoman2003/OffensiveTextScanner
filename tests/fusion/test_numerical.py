"""Numerical equivalence between fused Triton kernels and their PyTorch references.

Tolerances: atol=1e-4, rtol=1e-3 (matches the plan). Tests are skipped gracefully when no CUDA
device is visible so the suite still runs in CI.
"""
from __future__ import annotations

import math

import pytest
import torch

CUDA = torch.cuda.is_available()
TOL = dict(atol=1e-4, rtol=1e-3)


@pytest.mark.skipif(not CUDA, reason="fusion kernels require CUDA")
def test_fused_qkv_matches_reference():
    from ots_core.fusion.kernels.fused_qkv import fused_qkv, qkv_reference

    torch.manual_seed(0)
    B, N, H = 2, 128, 256
    x = torch.randn(B, N, H, device="cuda", dtype=torch.float32)
    wq = torch.randn(H, H, device="cuda") * 0.02
    wk = torch.randn(H, H, device="cuda") * 0.02
    wv = torch.randn(H, H, device="cuda") * 0.02
    bq = torch.randn(H, device="cuda") * 0.02
    bk = torch.randn(H, device="cuda") * 0.02
    bv = torch.randn(H, device="cuda") * 0.02

    w_qkv = torch.cat([wq, wk, wv], dim=0)
    b_qkv = torch.cat([bq, bk, bv], dim=0)

    q, k, v = fused_qkv(x, w_qkv, b_qkv)
    q_ref, k_ref, v_ref = qkv_reference(x, wq, wk, wv, bq, bk, bv)

    assert torch.allclose(q, q_ref, **TOL)
    assert torch.allclose(k, k_ref, **TOL)
    assert torch.allclose(v, v_ref, **TOL)


@pytest.mark.skipif(not CUDA, reason="fusion kernels require CUDA")
def test_fused_ln_linear_matches_reference():
    from ots_core.fusion.kernels.fused_ln_linear import fused_ln_linear

    torch.manual_seed(0)
    B, N, H, D = 2, 64, 128, 256
    x = torch.randn(B, N, H, device="cuda")
    gamma = torch.ones(H, device="cuda")
    beta = torch.zeros(H, device="cuda")
    w = torch.randn(D, H, device="cuda") * 0.02
    b = torch.randn(D, device="cuda") * 0.02

    out = fused_ln_linear(x, gamma, beta, w, b)
    ref = torch.nn.functional.linear(
        torch.nn.functional.layer_norm(x, (H,), gamma, beta), w, b
    )
    assert torch.allclose(out, ref, **TOL)


@pytest.mark.skipif(not CUDA, reason="fusion kernels require CUDA")
def test_fused_bias_gelu_matches_reference():
    from ots_core.fusion.kernels.fused_bias_gelu import fused_bias_gelu_dropout

    torch.manual_seed(0)
    x = torch.randn(4, 64, 256, device="cuda")
    b = torch.randn(256, device="cuda") * 0.02
    out = fused_bias_gelu_dropout(x, b, p=0.0, training=False)
    ref = torch.nn.functional.gelu(x + b, approximate="tanh")
    assert torch.allclose(out, ref, **TOL)


@pytest.mark.skipif(not CUDA, reason="fusion kernels require CUDA")
def test_flash_attention_matches_reference():
    from ots_core.fusion.kernels.fused_attention import flash_attention

    torch.manual_seed(0)
    B, H, N, D = 1, 4, 128, 64
    q = torch.randn(B, H, N, D, device="cuda")
    k = torch.randn(B, H, N, D, device="cuda")
    v = torch.randn(B, H, N, D, device="cuda")
    mask = torch.ones(B, N, device="cuda", dtype=torch.int32)

    out = flash_attention(q, k, v, attention_mask=mask)
    scale = 1.0 / math.sqrt(D)
    scores = (q @ k.transpose(-1, -2)) * scale
    ref = scores.softmax(dim=-1) @ v
    # flash-style has slightly looser tol due to online softmax normalisation
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-2)


def test_patch_embed_reference_cpu():
    """CPU-path sanity (runs in CI without CUDA)."""
    from ots_core.fusion.kernels.fused_patch_embed import fused_patch_embed
    import torch.nn.functional as F

    torch.manual_seed(0)
    B, C, H, W, hidden, P = 2, 3, 64, 64, 16, 32
    x = torch.randn(B, C, H, W)
    cw = torch.randn(hidden, C, P, P) * 0.02
    cb = torch.randn(hidden) * 0.02
    gamma = torch.ones(hidden)
    beta = torch.zeros(hidden)
    cls = torch.randn(1, 1, hidden)
    pos = torch.randn(1, 1 + (H // P) * (W // P), hidden)

    out = fused_patch_embed(x, cw, cb, gamma, beta, cls, pos)
    conv = F.conv2d(x, cw, cb, stride=P).flatten(2).transpose(1, 2)
    ref = torch.cat([cls.expand(B, -1, -1), F.layer_norm(conv, (hidden,), gamma, beta)], dim=1) + pos
    assert torch.allclose(out, ref, atol=1e-5)
