"""Custom Triton fusion kernels + module replacements for ViLT.

Two strategies (per the plan):
- **Horizontal**: merge parallel ops with a shared input (QKV projection, GLU Gate/Up).
- **Vertical**: merge sequential ops that pass through memory (LN+Linear, Bias+GELU+Dropout,
  attention score+softmax, Conv+LN+CLS/pos-add patch embedding).

Use ``patcher.patch_vilt(model)`` after loading to swap the relevant ``nn.Module``s in place.
"""

from ots_core.fusion.patcher import patch_vilt, restore_vilt

__all__ = ["patch_vilt", "restore_vilt"]


def triton_available() -> bool:
    """Return True iff we can import triton *and* a CUDA device is visible."""
    try:
        import triton  # noqa: F401
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False
