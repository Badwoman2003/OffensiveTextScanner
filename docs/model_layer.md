# 模型层实现详解

本文档详细说明 OTS 的模型层 — 从 OCR 前处理、ViLT 骨干改造、三阶段训练流水线、模态不平衡对抗，到 Triton 算子融合的完整实现路径。

## 1. 整体数据/推理流

```
┌─────────┐   ┌──────────────┐   ┌───────────────────┐   ┌────────────┐
│ 图片URL │──►│ 下载(asyncio)│──►│ RapidOCR 抽文     │──►│            │
└─────────┘   └──────────────┘   └───────────────────┘   │  ViLT      │
                                                          │  (fused)   │──► logits → softmax → label, prob
┌─────────┐                      ┌───────────────────┐   │            │
│ DOM文本 │─────────────────────►│ 拼接 [OCR] 分隔 ──┼──►│            │
└─────────┘                      └───────────────────┘   └────────────┘
```

核心组件：

| 组件                      | 文件                                                  | 作用                                             |
|---------------------------|-------------------------------------------------------|--------------------------------------------------|
| `RapidOCRExtractor`       | `ots_core/ocr/rapid_ocr.py`                           | 轻量 ONNX OCR，置信度+阅读顺序过滤              |
| `ViltForOffenseClassification` | `ots_core/models/vilt_multimodal.py`              | 中文化 ViLT + 分类头                            |
| `OffenseClassificationHead`, `ModalityHead` | `ots_core/models/heads.py`             | 主分类头 + GB 单模态辅助头                      |
| `FocalLoss`, `GradientBlendedLoss` | `ots_core/models/losses.py`                     | 损失函数                                         |
| `ScannerPipeline`         | `ots_core/inference/pipeline.py`                      | 端到端推理封装                                   |
| `DynamicBatcher`          | `ots_core/inference/batcher.py`                       | 时间窗 + 容量触发的动态批处理                   |

## 2. OCR 轻量化

**为什么不用 PaddleOCR**：模型体积 ~200MB，首次加载 >5s；Windows 下偶现 DLL 冲突；PaddlePaddle 与 PyTorch 同进程会抢占 CUDA context。

**替换方案**：`RapidOCR`，本质是 PP-OCRv4-mobile 的 ONNXRuntime 封装，CPU 单核 <100ms/图，总权重 <15MB。

封装接口（`ots_core/ocr/rapid_ocr.py`）：

```python
class RapidOCRExtractor:
    def __init__(self, min_conf: float = 0.6, min_chars: int = 2): ...
    def extract(self, image: str | Path | Image | ndarray) -> list[str]: ...
    def extract_detailed(self, image) -> list[OCRLine]: ...  # 带 bbox + conf
```

关键设计：

- **阅读顺序重排**：`_reading_order` 先按 y 坐标分行（容差 20px），再按 x 排序，保证多行 meme 文字的语义连续。
- **阈值双重过滤**：`min_conf` 过滤低置信 + `min_chars` 过滤垃圾字符（纯标点、单字噪声）。
- **Fallback 机制**：`rapidocr_onnxruntime` 未安装时 `_load_engine` 返回 `None`，`extract` 返回 `[]`，不抛异常 — CI 环境无需装 OCR。

## 3. ViLT 中文化改造

### 3.1 为什么选 ViLT

对比主流多模态路线：

| 模型        | 视觉 tokenizer     | 参数量 | 延迟（B=8，384×384） | 适合场景               |
|-------------|---------------------|--------|----------------------|------------------------|
| CLIP        | ViT（双塔）         | 150M+  | 中                   | 检索/zero-shot 分类    |
| BLIP        | ViT + Q-Former      | 370M   | 高                   | 生成/VQA               |
| **ViLT**    | 纯线性 Patch Embed  | 87M    | **低**               | **轻量分类**（本项目） |

ViLT 去掉了独立的视觉 CNN/ViT encoder，用一次 32×32 的 Conv2d 把 patch 投影到 token 空间后和文本 token 拼接，走同一个 Transformer。这让我们可以用**同一套 Triton kernel**同时加速图文两路。

### 3.2 三步中文化（`ots_core/models/vilt_multimodal.py`）

1. **Tokenizer 替换**：`BertTokenizerFast.from_pretrained("bert-base-chinese")`；保留 `ViltImageProcessor` 不变（图像归一化统计量照旧）。

2. **Embedding Warm-start**（`warm_start_embeddings`）：
   ```python
   old_mean = old_w.mean(dim=0, keepdim=True)            # 英文 embedding 均值
   new_emb.weight.data.copy_(old_mean.expand(new_vocab_size, -1))
   new_emb.weight.data[:copy_n] = old_w[:copy_n]          # 索引 overlap 部分直拷
   ```
   这样新 token 一开始就落在英文 embedding 的质心附近，比随机初始化稳定得多。

3. **可学习投影层**：`self.embed_projection = nn.Linear(H, H, bias=False)` 用 identity 初始化，让 CPT 阶段有能力把 warm-start 分布「搬」到更适合中文的位置，而不会破坏已经对齐的 patch embedding。

### 3.3 分类头（`heads.py`）

```
[CLS] → LayerNorm → Dropout → Linear(H→H) → GELU → Dropout → Linear(H→2)
```

保留多标签扩展位（`num_labels` 可配），未来可接 COLDataset 的细粒度标签。

### 3.4 OCR 文本注入

推理时 `ScannerPipeline._compose` 用 `[OCR]` 伪 token 把 OCR 文本拼到原文本尾部：

```python
"今天天气真好 [OCR] 某个 meme 图里的字"
```

训练时 `clean/ocr_prefetch.py` 对训练集做同样的 OCR 预抽取，保证训练和推理看到的 OCR 分布一致（这是保障 domain adaptation 的关键）。

## 4. 数据层

### 4.1 统一 Schema（`ots_core/data/schema.py`）

所有数据源都会归一化到同一套字段：

```python
@dataclass
class Sample:
    id: str
    text: str                       # DOM/原始文本
    ocr_text: str                   # 图像 OCR 文本
    image_path: str | None          # text-only 样本为 None
    label: int                      # 0 benign, 1 offensive
    source: Source                  # coldataset | hateful_memes | chmeme | ...
    modality_mask: ModalityMask     # text_only | image_only | both
```

这样下游 `MultimodalOffenseDataset` 无需区分数据源，通过 `modality_mask` 即可做模态感知的采样 / 评测。

### 4.2 数据管线

```
raw/ ── ingest/ ──► clean/(dedup + sanitize + ocr_prefetch) ──► augment/(text_render) ──► processed/*.parquet
```

- **`ingest/coldataset.py`**：直接 CSV → parquet，`modality_mask = text_only`。
- **`ingest/hateful_memes.py`**：Facebook 原始 JSONL；`--translator hf` 启用 `Helsinki-NLP/opus-mt-en-zh` 批译；OCR 文本保留英文原文作为对照（`ocr_text` 字段）。
- **`ingest/memes_scraper.py`**：可扩展接口 `MemeSource`，自带 `RobotsGate`（缓存 robots.txt 解析器）+ pHash 图片去重 + 并发下载。

### 4.3 去重与清洗

- **文本 SimHash**（`clean/dedup.py::_simhash_zh`）：按中文字符 3-gram 做 64-bit SimHash，按高 16 位做桶，桶内 Hamming ≤3 判重；**不依赖任何第三方**。
- **图像 pHash**（`imagehash.phash`）：Hamming ≤4 判重，典型 meme 变体重压缩/加水印仍能识别。
- **Sanitize**：`NFKC` 全半角归一化 → 剥 URL/@/#话题/连续标点 → 塌白空格。**不移除攻击性词汇**，否则会污染监督信号。

### 4.4 合成样本（`augment/text_render.py`）

把 COLDataset 文本用随机 HSV 柔和背景 + 随机字号渲染成 384×384 图片，标签继承原文本标签，`modality_mask = both`，`ocr_text = 原文本`。这批数据的作用是**让模型学会「图里的字」和「文本里的字」应该产生同一个决策**，有效缓解模态不平衡时模型只看文本的倾向。

### 4.5 模态感知评估（`data/evaluate.py`）

每次评估输出 4 个指标：

- `slices.text_only.f1_macro`
- `slices.image_only.f1_macro`
- `slices.both.f1_macro`
- **`modality_robustness_gap`** = max 切片 F1 − min 切片 F1

`gap` 过大说明模型偏科某一模态，需要上 Stage-C。

## 5. 三阶段训练流水线

### 5.1 Stage-A · 中文继续预训练（`training/stage_a_cpt.py`）

三任务联合优化：

| 任务 | 损失 | 权重 | 实现 |
|------|------|------|------|
| MLM  | CE，15% 掩码 | 1.0 | `_mlm_mask`：特殊 token 豁免，ignore_index=-100 |
| ITM  | CE，50% batch 内 shuffle 做负样本 | 1.0 | `_swap_for_itm`：对 `pixel_values` 置换索引 |
| WPA  | IPOT 最优传输 | 0.1 | `training/wpa.py::WordPatchAlignment` |

WPA 原论文用 Inexact Proximal OT 近似 Sinkhorn，把文本 token 和 patch token 的 cosine 距离矩阵做 20 次迭代的 OT 对齐。权重压到 0.1 是因为它更多是 **正则化**，不是主监督。

**关键代码（`wpa.py`）**：

```python
def _ipot(C, n_iter=20, beta=0.5):
    sigma = 1/N * ones;  T = uniform
    A = exp(-C/beta)
    for _ in range(n_iter):
        Q = A * T
        delta = 1/(M * Q @ sigma)
        sigma = 1/(N * Q.T @ delta)
        T = delta * Q * sigma.T
    return T  # 最优传输计划
```

### 5.2 Stage-B · 下游微调（`training/stage_b_finetune.py`）

- **Loss**：`FocalLoss(γ=2)` 替代 CE，缓解 benign 样本占多的不平衡；`--loss ce` 可切回 CE 做消融。
- **Sampler**：`StratifiedModalitySampler` 按 `(source, modality_mask, label)` 做 Cui et al. 的有效样本数加权：`w_k = (1-β) / (1-β^{n_k})`。
- **Modality Dropout**：`DataConfig.modality_dropout_p=0.15`，训练时 `__getitem__` 里以 15% 概率把某一模态置空，强制模型在单模态缺失时仍能出合理预测。
- **评估**：每个 epoch 结束跑一遍三切片 + overall，按 `overall.f1_macro` 保存 best checkpoint。

### 5.3 Stage-C · 模态不平衡专项（`training/stage_c_imbalance.py`）

**核心思路**：每个 batch 走三次 backbone（text-only / image-only / both 三个视图），分别产出 logits，用 `GradientBlendedLoss` 加权融合：

```
L = w_t · L(logits_text, y) + w_i · L(logits_image, y) + w_m · L(logits_multi, y)
```

权重不是超参，而是**动态求解**。每个 epoch 结束后：

1. 计算每路的 `StreamStats`：前后两次的 train_loss 差 Δtrain，val_loss 差 Δval。
2. `OGR = ((Δtrain - Δval)_+)² / Δval²`（过拟合程度系数）。
3. `w_k ∝ 1 / (OGR_k + ε)`，归一化。

实现见 `training/gradient_blending.py::solve_weights`。

**为什么有效**：当文本分支已经收敛但图像分支还在泛化，OGR 会告诉算法「多给图像分支权重」，避免模型偷懒只学文本捷径。

## 6. 算子融合层

落点 `ots_core/fusion/`，全部用 Triton 手写，每个 kernel 都配 PyTorch 参考实现（双重作用：CPU fallback + 数值对齐测试）。

### 6.1 Horizontal Fusing（同输入并行算子合并）

**Fused QKV Projection**（`fusion/kernels/fused_qkv.py`）：ViLT 每个 encoder block 有 3 次独立的 `nn.Linear(H, H)` 计算 Q/K/V，输入同一个 `X`。我们把权重沿输出维拼接：

```
W_qkv = [W_q; W_k; W_v]  ∈ ℝ^(3H × H)
b_qkv = [b_q; b_k; b_v]  ∈ ℝ^(3H)
[Q | K | V] = X @ W_qkv.T + b_qkv
```

一次 GEMM 吐出 3 倍宽度的输出，按列 split 得到 Q/K/V。Triton kernel `_qkv_fwd_kernel` 按 `(BLOCK_M × BLOCK_N)` tile 遍历，BLOCK_K=32 沿 K 维累加。收益：

- **3 次 kernel launch → 1 次**
- **3 次 W 读取 → 1 次**（W 的行在 L2 cache 连续）
- **输出写回 HBM 仍是 3H 宽，但只一次**

**Fused Gate/Up**（`fused_gate_up.py`）：为 GLU FFN 预留的同原理融合；ViLT 默认用普通 GELU FFN，这条路径在用户改造 FFN 为 GEGLU/SwiGLU 时才会激活。

### 6.2 Vertical Fusing（顺序算子串行合并）

**LayerNorm + Linear**（`fused_ln_linear.py`）：原流程是 LN 算完 mean/std 写回 HBM，再读出来做 matmul。融合后一个 kernel 内：

```
# Pass 1: 两次遍历 H 维累加 mean 和 m2（欢迎 two-pass 方差算法）
# Pass 2: 边归一化 x_hat = (x - mean) * rstd * γ + β, 边 acc += x_hat @ W.T
```

节省了中间 `(B·N·H)` 张量的 HBM 来回。

**Bias + GELU + Dropout**（`fused_bias_gelu.py`）：FFN 首层输出链 `Linear → +bias → GELU → Dropout` 的后三步合为一个 elementwise kernel。Dropout mask 用 `tl.rand(seed, offset)` 按需重生成，不落盘；seed 每次由 `torch.randint` 产生一次。GELU 用 tanh 近似：

```
gelu(x) = 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))
```

**Flash-style Attention**（`fused_attention.py`）：软最大化 + V 聚合一个 kernel 搞定。用 tile 分块 + online softmax 技巧：

```
m_i ← running row max
l_i ← running row sum (rescaled)
acc ← running Σ (p @ V) (rescaled)

for tile in K:
    qk = Q · K.T * scale
    if mask: qk[mask==0] = -inf
    m_new = max(m_i, max(qk, dim=-1))
    α = exp(m_i - m_new)          # 归一化因子修正
    p = exp(qk - m_new)
    l_i = l_i · α + sum(p, dim=-1)
    acc = acc · α + p @ V
    m_i = m_new

out = acc / l_i
```

对比标准实现，**省掉 N×N 的中间 score 矩阵**（ViLT N≈250，一个 head 就是 250×250=62500 个元素）。数值容差放宽到 `atol=1e-3, rtol=1e-2`，因为在线 softmax 的累计误差略高。

**Patch Embedding 融合**（`fused_patch_embed.py`）：ViLT 的 patch tokenizer 是 `Conv2d(stride=32,kernel=32)` → `flatten` → `transpose` → `LayerNorm` → 拼 `[CLS]` → 加 `pos_embedding`，五步合一个 kernel。目前落地了 PyTorch 参考版（`FUSED STEP` 注释处可直接替换为 Triton 代码）。

### 6.3 集成：`patcher.patch_vilt(model)`

```python
from ots_core.fusion import patch_vilt
model = ViltForOffenseClassification.load("checkpoints/final").cuda().eval()
stats = patch_vilt(model)
print(stats)
# FusionStats(attentions_patched=12, intermediates_patched=12, patch_embed_patched=True, skipped=[])
```

内部做法：遍历 `model.vilt.encoder.layer`，把每层的 `layer.attention.attention` 替换为 `FusedViltSelfAttention.from_hf(...)`（自动拼 QKV 权重），把 `layer.intermediate` 替换为 `FusedViltIntermediate`；再把 `embeddings` 的 patch embed 路径挂成 `FusedPatchEmbed`。idempotent、可报告 skipped 模块。

### 6.4 数值对齐与基准

- `tests/fusion/test_numerical.py`：CUDA 路径 `atol=1e-4, rtol=1e-3`；Flash Attention `atol=1e-3, rtol=1e-2`；patch embed 走 CPU 参考路径 `atol=1e-5`。无 CUDA 时自动 skip。
- `benchmarks/bench_fusion.py`：`torch.cuda.Event` 测量 200 次迭代 + 10 次 warmup；`torch.profiler` 统计 CUDA op 数；`torch.cuda.max_memory_allocated` 统计峰值显存；自动打印 `P50 提速%` / `显存下降%` / `op 数下降%`。

### 6.5 何时不开融合

`ots_core/config.py::OTSSettings.use_fused_kernels = True` 控制，默认开启。如果：

- 目标是数值完全可比的 baseline 评测 → 关掉
- 环境没 CUDA/Triton → `ots_core.fusion.triton_available()` 返回 False，`ScannerPipeline` 自动跳过融合

## 7. 检查点管理

`ViltForOffenseClassification.save(dir)` 写出：

```
<dir>/
├── pytorch_model.bin             # state_dict
├── config.json                   # ViltConfig
├── tokenizer.json / vocab.txt    # bert-base-chinese 词表
├── preprocessor_config.json      # ViltImageProcessor 参数
└── special_tokens_map.json
```

`ViltForOffenseClassification.load(dir)` 用相同格式加载。与 HuggingFace `from_pretrained` 布局兼容，可直接 push 到 Hub。

## 8. 导出到其他推理栈

`scripts/export_onnx.py`：

```bash
# ONNX（跨运行时通用，TensorRT / ONNX Runtime 可直接吃）
python scripts/export_onnx.py --ckpt checkpoints/final --out exports/ots.onnx

# TorchScript + 融合 kernel（最快，但只能 PyTorch 运行时）
python scripts/export_onnx.py --ckpt checkpoints/final --out exports/ots.pt --fuse
```

注意 ONNX 导出不包含 Triton fused kernel（ONNX 只覆盖标准 op 集）；TorchScript 走 trace，会把融合后的 module 固化。
