"""Loss functions: Focal loss + gradient-blended multimodal loss."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Binary/multi-class focal loss (Lin et al., 2017)."""

    def __init__(self, gamma: float = 2.0, alpha: float | torch.Tensor | None = None, reduction: str = "mean") -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=-1)
        logp_t = logp.gather(1, target.unsqueeze(1)).squeeze(1)
        p_t = logp_t.exp()
        loss = -((1 - p_t) ** self.gamma) * logp_t

        if self.alpha is not None:
            alpha = self.alpha
            if isinstance(alpha, torch.Tensor):
                alpha = alpha.to(logits.device)
                loss = loss * alpha.gather(0, target)
            else:
                loss = loss * alpha

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class GradientBlendedLoss(nn.Module):
    """Combine three parallel losses (text, image, multimodal) with weights computed by the
    Gradient-Blending algorithm. Weights are not learned from backprop; the trainer updates
    them out-of-band after each epoch by calling ``update_weights``.
    """

    def __init__(self, base_loss: nn.Module) -> None:
        super().__init__()
        self.base = base_loss
        self.register_buffer("w_text", torch.tensor(1.0 / 3))
        self.register_buffer("w_image", torch.tensor(1.0 / 3))
        self.register_buffer("w_multi", torch.tensor(1.0 / 3))

    def update_weights(self, w_text: float, w_image: float, w_multi: float) -> None:
        s = max(1e-6, w_text + w_image + w_multi)
        self.w_text.fill_(w_text / s)
        self.w_image.fill_(w_image / s)
        self.w_multi.fill_(w_multi / s)

    def forward(
        self,
        logits_text: torch.Tensor,
        logits_image: torch.Tensor,
        logits_multi: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        l_t = self.base(logits_text, target)
        l_i = self.base(logits_image, target)
        l_m = self.base(logits_multi, target)
        total = self.w_text * l_t + self.w_image * l_i + self.w_multi * l_m
        return {
            "loss": total,
            "loss_text": l_t.detach(),
            "loss_image": l_i.detach(),
            "loss_multi": l_m.detach(),
        }
