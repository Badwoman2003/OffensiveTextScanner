"""Gradient-Blending weight solver (Wang et al., CVPR'20 — "What Makes Training Multi-Modal...").

We approximate Overfitting-to-Generalization Ratio (OGR) for each stream by tracking the
validation and training losses across two consecutive checkpoints. The new weights minimise
the estimated sum of over-fitting contributions.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StreamStats:
    train_loss_prev: float
    train_loss_now: float
    val_loss_prev: float
    val_loss_now: float

    @property
    def ogr(self) -> float:
        """OGR_n = (ΔL_val) / (ΔL_train)^2 in the original paper's second-order form.

        We clamp numerator/denominator to keep the weights sane when a stream is already
        saturated.
        """
        dl_tr = self.train_loss_prev - self.train_loss_now
        dl_val = self.val_loss_prev - self.val_loss_now
        # a stream that *increased* its val loss while training loss fell is over-fitting.
        over_fit = max(0.0, dl_tr - dl_val)
        gen = max(1e-6, dl_val)
        return (over_fit ** 2) / (gen ** 2 + 1e-6)


def solve_weights(text: StreamStats, image: StreamStats, multi: StreamStats) -> tuple[float, float, float]:
    """Return (w_text, w_image, w_multi) normalised so they sum to 1.

    Each weight is proportional to the *inverse* of that stream's OGR, so streams that over-fit
    less get upweighted. A uniform prior (ε=0.05) prevents total collapse.
    """
    eps = 0.05
    inv = [1.0 / (s.ogr + eps) for s in (text, image, multi)]
    total = sum(inv)
    return (inv[0] / total, inv[1] / total, inv[2] / total)
