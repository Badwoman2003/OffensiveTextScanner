# Operator Fusion

OTS implements two classes of kernel fusion against the ViLT backbone, producing a single
replacement model that is numerically close to the stock HuggingFace implementation but runs
with fewer CUDA kernel launches and less HBM traffic.

## Horizontal fusing — parallel ops, same input

| Pattern                    | File                                                  | Notes                                                                 |
|----------------------------|-------------------------------------------------------|-----------------------------------------------------------------------|
| Fused Q · K · V projection | `ots_core/fusion/kernels/fused_qkv.py`                | Stacks `W_q,W_k,W_v` along the output dim; one GEMM instead of three. |
| Fused Gate / Up projection | `ots_core/fusion/kernels/fused_gate_up.py`            | Useful when swapping ViLT's FFN for GEGLU/SwiGLU.                      |

## Vertical fusing — sequential ops, memory chain

| Pattern                                       | File                                                    | Savings                                         |
|-----------------------------------------------|---------------------------------------------------------|-------------------------------------------------|
| LayerNorm + Linear                            | `ots_core/fusion/kernels/fused_ln_linear.py`            | Avoids writing the normalised activation to HBM |
| Bias + GELU + Dropout                         | `ots_core/fusion/kernels/fused_bias_gelu.py`            | Three elementwise passes → one                   |
| Attention score + softmax + value aggregation | `ots_core/fusion/kernels/fused_attention.py`            | Flash-Attention-style tiled online softmax       |
| Conv+LN+[CLS]+pos for patch embedding         | `ots_core/fusion/kernels/fused_patch_embed.py`          | One pass over input image                        |

## Integration — `patch_vilt`

```python
from ots_core.fusion import patch_vilt
from ots_core.models.vilt_multimodal import ViltForOffenseClassification

model = ViltForOffenseClassification.load("checkpoints/vilt-offense-final").cuda().eval()
stats = patch_vilt(model)
print(stats)  # attentions_patched=12, intermediates_patched=12, patch_embed_patched=True
```

## Numerical validation

Unit tests in `tests/fusion/test_numerical.py` check every kernel within `atol=1e-4, rtol=1e-3`
(Flash attention slightly looser due to online normalisation).

## Benchmark

```bash
python benchmarks/bench_fusion.py --batch 8 --seq 128 --iters 200
```

Outputs `benchmarks/results/fusion_<ts>.json`. Plan targets: kernel count ↓ ≥ 30 %, P50 latency
↓ ≥ 20 %, peak HBM ↓ noticeably.
