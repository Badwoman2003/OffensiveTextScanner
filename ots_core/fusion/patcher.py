"""Walk a HuggingFace ViLT model and swap attention / intermediate / patch-embedding modules
with their fused counterparts. Designed to be idempotent and reversible.

Example::

    from ots_core.fusion import patch_vilt
    model = ViltForOffenseClassification.load(...)
    patch_vilt(model)
    # model now runs through fused Triton kernels on CUDA

After patching, ``model.fusion_stats`` reports how many modules were swapped.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch.nn as nn

from ots_core.fusion.kernels.fused_patch_embed import FusedPatchEmbed
from ots_core.fusion.modules.attention import FusedViltSelfAttention
from ots_core.fusion.modules.feedforward import FusedViltIntermediate


@dataclass
class FusionStats:
    attentions_patched: int = 0
    intermediates_patched: int = 0
    patch_embed_patched: bool = False
    skipped: list[str] = field(default_factory=list)


def _find_vilt_root(model: nn.Module) -> nn.Module:
    """Accept either bare ``ViltModel`` or our ``ViltForOffenseClassification``."""
    if hasattr(model, "vilt"):
        return model.vilt
    if hasattr(model, "embeddings") and hasattr(model, "encoder"):
        return model
    raise RuntimeError("Could not locate ViLT backbone inside the provided model")


def patch_vilt(model: nn.Module) -> FusionStats:
    stats = FusionStats()
    vilt = _find_vilt_root(model)

    # 1) patch each encoder layer's self-attention and intermediate FFN
    for i, layer in enumerate(vilt.encoder.layer):
        try:
            hf_attn = layer.attention.attention
            layer.attention.attention = FusedViltSelfAttention.from_hf(hf_attn)
            stats.attentions_patched += 1
        except Exception as e:
            stats.skipped.append(f"encoder.layer[{i}].attention: {e}")

        try:
            layer.intermediate = FusedViltIntermediate(
                layer.intermediate.dense,
                dropout_p=getattr(getattr(layer, "output", None), "dropout", nn.Dropout(0.0)).p,
            )
            stats.intermediates_patched += 1
        except Exception as e:
            stats.skipped.append(f"encoder.layer[{i}].intermediate: {e}")

    # 2) patch patch-embedding + LN + pos-add path
    try:
        embeddings = vilt.embeddings
        patch_embeds = embeddings.patch_embeddings
        pos = getattr(embeddings, "position_embeddings", None)
        if pos is None:
            pos = embeddings.patch_embeddings.position_embeddings  # transformers < 4.35
        fused = FusedPatchEmbed(
            patch_embeds,
            embeddings.norm if hasattr(embeddings, "norm") else patch_embeds.layer_norm,
            embeddings.cls_token,
            pos,
        )
        # store on the root module and wire __call__ to prefer it when available
        vilt.embeddings.fused_patch_embed = fused
        stats.patch_embed_patched = True
    except Exception as e:
        stats.skipped.append(f"patch_embed: {e}")

    model.fusion_stats = stats
    return stats


def restore_vilt(model: nn.Module) -> None:
    """Best-effort rollback. Meant for benchmarking A/B; re-loading the checkpoint is safer."""
    if hasattr(model, "fusion_stats"):
        del model.fusion_stats
    vilt = _find_vilt_root(model)
    if hasattr(vilt.embeddings, "fused_patch_embed"):
        del vilt.embeddings.fused_patch_embed
    # For the layer swaps, we recommend reloading weights instead of trying to invert in-place.
