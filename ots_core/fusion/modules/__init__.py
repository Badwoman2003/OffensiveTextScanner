"""High-level nn.Modules that wire the fused kernels to the HuggingFace ViLT layers."""

from ots_core.fusion.modules.attention import FusedViltSelfAttention
from ots_core.fusion.modules.feedforward import FusedViltIntermediate, FusedViltOutput

__all__ = ["FusedViltSelfAttention", "FusedViltIntermediate", "FusedViltOutput"]
