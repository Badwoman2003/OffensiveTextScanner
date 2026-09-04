# OTS Architecture (v0.2)

```
                    ┌─────────────────────────┐
                    │   Chrome Extension      │
                    │  (React + Vite + TS)    │
                    │  popup / options /      │
                    │  content / background   │
                    └───────────┬─────────────┘
                                │  HTTPS + Bearer token
                                ▼
                    ┌─────────────────────────┐
                    │   FastAPI Gateway       │
                    │  - auth / CORS          │
                    │  - rate-limit (Redis)   │
                    │  - SSE / WebSocket      │
                    │  - Prometheus / OTel    │
                    └───────────┬─────────────┘
                                │  Celery (Redis broker)
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
        ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
        │ worker-ocr  │ │ worker-cpu  │ │ worker-gpu  │
        │ RapidOCR    │ │ ViLT (cpu)  │ │ ViLT (cuda) │
        │             │ │             │ │ +Triton fuse│
        └─────────────┘ └─────────────┘ └──────┬──────┘
                                               │ results
                                               ▼
                                         ┌───────────┐
                                         │  Redis    │
                                         │  status + │
                                         │  result   │
                                         └───────────┘
```

## Stack-to-requirement mapping

| Requirement                              | Module                                                             |
|------------------------------------------|--------------------------------------------------------------------|
| Multimodal backbone (ViLT)               | `ots_core/models/vilt_multimodal.py`                               |
| Lightweight OCR                          | `ots_core/ocr/rapid_ocr.py`                                        |
| Full-lifecycle training pipeline         | `ots_core/training/stage_{a,b,c}_*.py`                             |
| Dataset: COLDataset + Hateful Memes + memes | `ots_core/data/ingest/*.py`, `ots_core/data/dataset.py`        |
| Modality-imbalance mitigation            | `ots_core/models/losses.py` + `ots_core/training/gradient_blending.py` |
| Horizontal operator fusion               | `ots_core/fusion/kernels/fused_qkv.py`, `fused_gate_up.py`         |
| Vertical operator fusion                 | `ots_core/fusion/kernels/fused_ln_linear.py`, `fused_bias_gelu.py`, `fused_attention.py`, `fused_patch_embed.py` |
| High-concurrency backend                 | `backend/app/main.py` + `backend/worker/*`                         |
| Revamped extension                       | `extension/*`                                                       |

## Request lifecycle

1. User clicks the toolbar button → `background.ts` messages the active tab's content script.
2. Content script walks the DOM (`TreeWalker` + `MutationObserver`) and returns text + image URLs.
3. Background posts `POST /api/v1/scan`. API enqueues a Celery task and returns `job_id`.
4. Worker process loads the ViLT pipeline once, runs OCR + dynamic-batched inference, writes
   the final `ScanResponse` JSON into Redis (`ots:result:<job_id>`).
5. Extension polls `GET /api/v1/scan/<job_id>/status`, then fetches `GET /api/v1/scan/<job_id>`
   when status ∈ {success, failed}. SSE and WebSocket endpoints are also available.
6. Content script renders the floating card + highlights offensive spans; users may flag false
   positives, which are batched into `ots:feedback` for periodic retraining.
