"""Word-Patch Alignment (WPA) loss used in ViLT pretraining.

Computes an Inexact Proximal Optimal Transport (IPOT) distance between the text token hidden
states and the image patch hidden states. Following the official ViLT implementation we only
do a handful of Sinkhorn-style iterations — exact OT is not needed for a regulariser.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _ipot(C: torch.Tensor, n_iter: int = 20, beta: float = 0.5) -> torch.Tensor:
    """Inexact Proximal Optimal Transport. C: (B, M, N) cost matrix. Returns transport plan T."""
    B, M, N = C.shape
    sigma = torch.full((B, N, 1), 1.0 / N, device=C.device, dtype=C.dtype)
    T = torch.ones(B, M, N, device=C.device, dtype=C.dtype) / (M * N)
    A = torch.exp(-C / beta)
    for _ in range(n_iter):
        Q = A * T
        delta = 1.0 / (M * (Q @ sigma).clamp_min(1e-6))
        sigma = 1.0 / (N * (Q.transpose(1, 2) @ delta).clamp_min(1e-6))
        T = delta * Q * sigma.transpose(1, 2)
    return T


class WordPatchAlignment(nn.Module):
    def __init__(self, n_iter: int = 20, beta: float = 0.5) -> None:
        super().__init__()
        self.n_iter = n_iter
        self.beta = beta

    def forward(
        self,
        text_hidden: torch.Tensor,          # (B, Ltxt, H)
        image_hidden: torch.Tensor,         # (B, Limg, H)
        attention_mask: torch.Tensor,       # (B, Ltxt) — zero out pads
    ) -> torch.Tensor:
        t = torch.nn.functional.normalize(text_hidden, dim=-1)
        v = torch.nn.functional.normalize(image_hidden, dim=-1)
        C = 1.0 - torch.bmm(t, v.transpose(1, 2))                         # (B, Ltxt, Limg) cosine distance
        with torch.no_grad():
            T = _ipot(C, self.n_iter, self.beta)
        loss = (T * C).sum(dim=(1, 2))
        if attention_mask is not None:
            counts = attention_mask.sum(dim=1).clamp_min(1)
            loss = loss / counts
        return loss.mean()
