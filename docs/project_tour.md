# Project Tour

A complete tour of the OTS v0.2 monorepo, grouped by subsystem.

## Runtime stack

| Layer       | Tech                                                            |
|-------------|-----------------------------------------------------------------|
| Frontend    | Chrome MV3 extension · React 18 · Vite · TypeScript · Tailwind  |
| API         | FastAPI · Pydantic v2 · Uvicorn · Prometheus · OpenTelemetry    |
| Orchestration | Celery · Redis                                                |
| Inference   | PyTorch 2.2 · Triton 2.2 · RapidOCR                             |
| Training    | HuggingFace Transformers · Datasets · AMP                       |

## Package map

```
ots_core/
├── config.py                       # Pydantic settings, env-driven
├── data/
│   ├── schema.py                   # canonical Sample dataclass + arrow schema
│   ├── ingest/
│   │   ├── coldataset.py           # COLDataset → parquet
│   │   ├── hateful_memes.py        # Hateful Memes + EN→ZH translation
│   │   └── memes_scraper.py        # robots-aware Chinese meme scraper
│   ├── clean/
│   │   ├── dedup.py                # pHash image dedup + SimHash text dedup
│   │   ├── sanitize.py             # URL / mention / topic / punct normalisation
│   │   └── ocr_prefetch.py         # precompute OCR so train == infer distribution
│   ├── augment/text_render.py      # render COLDataset text onto neutral backgrounds
│   ├── dataset.py                  # MultimodalOffenseDataset + StratifiedModalitySampler
│   ├── collator.py                 # ViLT-aware batch collator
│   └── evaluate.py                 # text-only / image-only / both slice F1 + gap
├── ocr/rapid_ocr.py                # RapidOCR wrapper with conf / min-char filters
├── models/
│   ├── vilt_multimodal.py          # ViltForOffenseClassification + zh tokenizer swap
│   ├── heads.py                    # classification + modality heads
│   └── losses.py                   # FocalLoss, GradientBlendedLoss
├── training/
│   ├── stage_a_cpt.py              # MLM + ITM + WPA continued pretraining
│   ├── stage_b_finetune.py         # supervised FT with focal / sampler
│   ├── stage_c_imbalance.py        # Gradient-Blending specialist
│   ├── gradient_blending.py        # StreamStats + solve_weights
│   └── wpa.py                      # IPOT word-patch alignment loss
├── fusion/
│   ├── patcher.py                  # monkey-patch HF ViLT with fused modules
│   ├── kernels/                    # Triton kernel implementations
│   │   ├── fused_qkv.py            # horizontal: Q|K|V in one GEMM
│   │   ├── fused_gate_up.py        # horizontal: GLU gate+up
│   │   ├── fused_ln_linear.py      # vertical: LN + Linear
│   │   ├── fused_bias_gelu.py      # vertical: bias + GELU + dropout
│   │   ├── fused_attention.py      # vertical: flash-style softmax+V
│   │   └── fused_patch_embed.py    # vertical: Conv+LN+CLS+pos
│   └── modules/                    # nn.Module wrappers that host the kernels
└── inference/
    ├── pipeline.py                 # ScannerPipeline (OCR → ViLT → probs)
    └── batcher.py                  # DynamicBatcher (10 ms window, up to 16)

backend/
├── app/
│   ├── main.py                     # FastAPI factory, metrics, CORS, OTel
│   ├── deps.py                     # Redis, bearer-token auth
│   ├── middleware/rate_limit.py    # Redis-backed per-client rate limiter
│   └── routers/scan.py             # POST /scan, GET /scan/{id}, SSE, WebSocket
├── schemas/scan.py                 # request/response pydantic models
├── worker/
│   ├── celery_app.py               # queue routing (ocr / cpu_inference / gpu_inference)
│   └── tasks/
│       ├── ocr_task.py             # async image download + RapidOCR batch
│       └── inference_task.py       # runs pipeline through DynamicBatcher
└── docker/
    ├── backend.Dockerfile
    └── worker.Dockerfile

extension/
├── manifest.json                   # MV3 with configurable host permissions
├── vite.config.ts                  # @crxjs/vite-plugin bundler
├── tailwind.config.js
└── src/
    ├── background/background.ts    # service worker, API orchestration
    ├── content/content.ts          # MutationObserver + IntersectionObserver + floating card
    ├── popup/                      # React popup (scan button + latest scan summary)
    ├── options/                    # React settings + history panel
    ├── lib/                        # api / storage / types
    └── components/                 # reusable Badge / ResultCard

benchmarks/
├── bench_fusion.py                 # torch.cuda.Event + profiler, A/B fused vs baseline
└── bench_api.py                    # locust user classes for text-only + multimodal

scripts/
├── dev_up.sh / .ps1                # docker compose for local dev
├── export_onnx.py                  # ONNX + TorchScript export
└── build_baseline_report.py        # aggregate histories into docs/baseline_report.md

tests/
├── fusion/test_numerical.py        # atol=1e-4/rtol=1e-3 vs PyTorch reference
├── data/test_pipeline.py           # sanitize / dedup / sampler
└── backend/test_schemas.py         # pydantic validation

legacy/
└── OffensiveTextScanner/           # original Flask + BERT + PaddleOCR baseline
```

## Commands cheatsheet

```bash
# Install Python deps
pip install -e ".[backend,triton,dev]"

# Train (happy path)
python -m ots_core.data.ingest.coldataset --root data/raw/COLDataset --out data/processed/coldataset
python -m ots_core.data.ingest.hateful_memes --root data/raw/hateful_memes --out data/processed/hateful_memes --translator hf
python -m ots_core.training.stage_a_cpt --pairs data/processed/cc3m_zh.parquet --out checkpoints/vilt-zh-cpt --fp16
python -m ots_core.training.stage_b_finetune --train data/processed/train_mixed.parquet --val data/processed/val_mixed.parquet --init checkpoints/vilt-zh-cpt/final --out checkpoints/stageB --fp16
python -m ots_core.training.stage_c_imbalance --train data/processed/train_mixed.parquet --val data/processed/val_mixed.parquet --init checkpoints/stageB/best --out checkpoints/final

# Benchmark fusion
python benchmarks/bench_fusion.py --batch 8 --seq 128

# Run services
docker compose up -d
locust -f benchmarks/bench_api.py --headless -u 100 -r 10 -t 2m

# Build extension
cd extension && npm install && npm run build
```
