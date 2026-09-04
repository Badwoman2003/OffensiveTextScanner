# OTS · Offensive Text Scanner

> 基于 ViLT 多模态模型的网页攻击性内容检测系统 · 浏览器扩展 · 算子融合 · 高并发推理后端

OTS（Offensive Text Scanner）是一套端到端的多模态攻击性内容检测系统。它由一个 Chrome 扩展、一个 FastAPI + Celery 后端，和一个基于 ViLT 的多模态模型组成，能在用户浏览任意网页时一键扫描页面上的文字与图片，识别其中是否包含攻击性/仇恨言论。

```
┌──────────────┐  HTTPS  ┌─────────────┐  Celery  ┌────────────────────┐
│ Chrome 扩展  │ ──────► │  FastAPI    │ ───────► │ Worker (ViLT+Triton│
│ React+TS+TW  │ ◄────── │  + Redis    │ ◄─────── │ fused kernels)     │
└──────────────┘   JSON  └─────────────┘  result  └────────────────────┘
```

## 功能亮点

- **真正的多模态**：用 ViLT 单流 Transformer 直接处理 (图像 patch + 文本 token)，不是两个单模态模型拼接。
- **OCR 回流**：RapidOCR 抽图内字后直接拼进 ViLT 文本端，图中文字与 DOM 文字一起参与决策。
- **中文适配**：英文 ViLT 继续预训练（CPT） + bert-base-chinese tokenizer + 中文 meme 微调。
- **模态不平衡处理**：Gradient-Blending 动态权重 + Modality Dropout + Focal Loss + 分层重采样。
- **Triton 算子融合**：水平融合（Fused QKV）+ 垂直融合（LN+Linear、Flash-style Attention、Bias+GELU+Dropout、Patch Embedding），推理延迟可降 ≥20%，显存峰值显著下降。
- **高并发后端**：FastAPI + Celery + Redis + 动态批处理（10ms 窗，batch=16），支持轮询/SSE/WebSocket 三种结果获取。
- **现代扩展**：React + Vite + TS + Tailwind，MutationObserver 覆盖 SPA，反馈回传形成闭环训练数据。

## 推荐硬件配置

| 场景 | CPU | GPU | 内存 | 说明 |
|------|-----|-----|------|------|
| 纯推理（单模态文本） | 任意 4 核 | 可选 | 8GB+ | CPU 即可，~200ms/样本 |
| 推理（多模态 + 融合 kernel） | 8 核 | **RTX 3060 12GB 及以上**（推荐 RTX 4090 / A6000） | 16GB+ | 融合 kernel 需 Compute Capability ≥ 7.5 |
| Stage-B/C 微调 | 16 核 | **单张 A100 40GB** 或 2× RTX 3090 | 64GB+ | fp16 + bs=32，全量微调约 4–6h |
| Stage-A CPT | 32 核 | **多卡 A100 80GB × 4** 或更多 | 256GB+ | 视语料规模 1–10M 对，耗时 2–5 天 |

> **Triton 要求**：NVIDIA GPU + CUDA 12.x + Compute Capability ≥ 7.5（Turing / Ampere / Hopper）。AMD ROCm 暂未验证。
>
> **不装 GPU 也能跑**：所有融合 kernel 都有 PyTorch 参考 fallback，纯 CPU 推理正确但不开启融合优化。

## 环境要求

| 组件 | 版本 |
|------|------|
| Python | ≥ 3.10 |
| PyTorch | ≥ 2.2 |
| CUDA | 12.1 推荐（PyTorch 官方构建匹配） |
| Triton | ≥ 2.2（随 PyTorch 自带） |
| Redis | ≥ 7 |
| Node.js | ≥ 20（构建扩展用） |
| Docker | ≥ 24（可选，一键起后端用） |

## 快速开始

### 1. 克隆并安装 Python 依赖

```bash
git clone https://github.com/<your-org>/OTS.git
cd OTS

# 基础依赖
pip install -e .

# 带后端 + Triton 融合 + 开发工具
pip install -e ".[backend,triton,dev]"
```

### 2. 一键拉起后端（Docker Compose）

```bash
cp .env.example .env
docker compose up -d redis backend worker-cpu worker-gpu flower
```

服务将监听：

- `http://localhost:8080`  — FastAPI 网关
- `http://localhost:5555`  — Flower（Celery 监控面板）
- `http://localhost:6379`  — Redis

健康检查：`curl http://localhost:8080/healthz`。

### 3. 构建并加载浏览器扩展

```bash
cd extension
npm install
npm run build
```

Chrome → `chrome://extensions` → 开启「开发者模式」→「加载已解压的扩展程序」→ 选择 `extension/dist`。

第一次使用时点开扩展图标的「Settings」，配置后端 URL（默认 `http://127.0.0.1:8080`）、API Token（默认 `dev-token-change-me`，生产请修改 `.env`）。

### 4. 扫描任意网页

点击扩展图标 → "Scan this page" → 右下角弹出悬浮卡片显示结果，攻击性文本会被红色边框高亮，每条可一键「Mark as false positive」回传反馈。

## 训练模型（可选）

```bash
# Stage-A 中文继续预训练
python -m ots_core.training.stage_a_cpt \
    --pairs data/processed/cc3m_zh.parquet \
    --out checkpoints/vilt-zh-cpt --fp16

# Stage-B 下游微调
python -m ots_core.training.stage_b_finetune \
    --train data/processed/train_mixed.parquet \
    --val   data/processed/val_mixed.parquet \
    --init  checkpoints/vilt-zh-cpt/final \
    --out   checkpoints/stageB --fp16

# Stage-C 模态不平衡专项训练
python -m ots_core.training.stage_c_imbalance \
    --train data/processed/train_mixed.parquet \
    --val   data/processed/val_mixed.parquet \
    --init  checkpoints/stageB/best \
    --out   checkpoints/final
```

详细训练流程、数据集准备、超参说明见 [`docs/model_layer.md`](model_layer.md)。

## 性能基准

```bash
# 算子融合 A/B 基准
python benchmarks/bench_fusion.py --batch 8 --seq 128 --iters 200

# 端到端压测
locust -f benchmarks/bench_api.py --headless -u 100 -r 10 -t 2m --host http://localhost:8080
```

目标指标：

- 单次推理 **kernel 数 ↓ ≥ 30%**，延迟 **↓ ≥ 20%**
- 单 GPU 机器 **≥ 200 QPS**（小 batch 文本）
- 多模态请求端到端 **P95 < 800ms**

## 文档导航

| 文档 | 内容 |
|------|------|
| [`model_layer.md`](model_layer.md) | ViLT 改造、三阶段训练、算子融合、模态不平衡实现细节 |
| [`backend_layer.md`](backend_layer.md) | FastAPI + Celery + Redis 架构、动态批处理、限流鉴权 |
| [`extension_layer.md`](extension_layer.md) | Chrome MV3 扩展架构、SPA 适配、悬浮卡片、反馈闭环 |
| [`architecture.md`](architecture.md) | 整体架构总览 + 请求生命周期 |
| [`training.md`](training.md) | 训练脚本 CLI 参考 |
| [`fusion.md`](fusion.md) | Triton kernel 使用与基准 |
| [`project_tour.md`](project_tour.md) | 完整目录与模块映射 |

## 许可证

MIT License · 数据集仅限研究使用，遵守各数据源原始条款。
