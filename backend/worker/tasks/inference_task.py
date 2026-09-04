"""Celery task that runs OCR (local call, no extra hop) + ViLT inference via ``DynamicBatcher``.

The batcher is created lazily on first call and kept alive for the lifetime of the worker
process so we don't pay model-load cost per task.
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
from typing import Any

import httpx
import redis
from PIL import Image

from backend.schemas.scan import ScanResponse, ScanResultRow
from backend.worker.celery_app import app
from backend.worker.tasks.ocr_task import _fetch_all
from ots_core.config import get_celery_settings
from ots_core.inference.batcher import DynamicBatcher
from ots_core.inference.pipeline import ScanItem, ScannerPipeline

log = logging.getLogger("ots.infer")

_pipeline: ScannerPipeline | None = None
_batcher: DynamicBatcher | None = None
_redis: redis.Redis | None = None


def _lazy_redis() -> redis.Redis:
    global _redis
    if _redis is None:
        _redis = redis.Redis.from_url(get_celery_settings().redis_url, decode_responses=True)
    return _redis


def _lazy_pipeline() -> ScannerPipeline:
    global _pipeline, _batcher
    if _pipeline is None:
        _pipeline = ScannerPipeline()
        _batcher = DynamicBatcher(_pipeline.batch_scan, max_batch=16, max_wait_ms=10)
    return _pipeline


def _download_images(urls: list[str]) -> list[Image.Image | None]:
    return asyncio.run(_fetch_all(urls))


def _build_items(text_blocks: list[str], images: list[Image.Image | None]) -> list[ScanItem]:
    items: list[ScanItem] = []
    for text in text_blocks:
        items.append(ScanItem(text=text, image=None))
    for img in images:
        items.append(ScanItem(text="", image=img))
    return items


def _set_status(job_id: str, status: str, progress: float = 0.0) -> None:
    r = _lazy_redis()
    r.set(f"ots:status:{job_id}", status, ex=3600)
    r.set(f"ots:progress:{job_id}", f"{progress:.3f}", ex=3600)


@app.task(name="ots.scan", bind=True)
def scan_task(self, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    _set_status(job_id, "running", 0.05)
    _lazy_pipeline()  # ensure model is warm

    text_blocks: list[str] = payload.get("text_blocks", [])
    image_urls: list[str] = payload.get("image_urls", [])

    try:
        images = _download_images(image_urls) if image_urls else []
        _set_status(job_id, "running", 0.35)

        items = _build_items(text_blocks, images)
        if not items:
            response = ScanResponse(job_id=job_id, status="success", results=[], offensive_count=0)
        else:
            futures = [_batcher.submit(it) for it in items]  # type: ignore[arg-type]
            results = [f.result(timeout=60) for f in futures]
            rows = [
                ScanResultRow(
                    text=r.text or r.ocr_text,
                    ocr_text=r.ocr_text,
                    label=r.label,
                    prob_offensive=r.prob_offensive,
                    modality_used=r.modality_used,
                )
                for r in results
            ]
            response = ScanResponse(
                job_id=job_id,
                status="success",
                results=rows,
                offensive_count=sum(1 for r in rows if r.label == 1),
                summary={
                    "n_text": sum(1 for r in rows if r.modality_used == "text_only"),
                    "n_image": sum(1 for r in rows if r.modality_used == "image_only"),
                    "n_both": sum(1 for r in rows if r.modality_used == "both"),
                },
            )

        _lazy_redis().set(
            f"ots:result:{job_id}",
            response.model_dump_json(),
            ex=3600,
        )
        _set_status(job_id, "success", 1.0)
        return {"job_id": job_id, "offensive_count": response.offensive_count}
    except Exception as exc:
        log.exception("scan_task failed: %s", exc)
        response = ScanResponse(job_id=job_id, status="failed", results=[], offensive_count=0,
                                summary={"error": str(exc)})
        _lazy_redis().set(f"ots:result:{job_id}", response.model_dump_json(), ex=3600)
        _set_status(job_id, "failed", 1.0)
        raise
