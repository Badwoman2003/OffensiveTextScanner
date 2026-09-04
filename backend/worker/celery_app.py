"""Celery application shared by CPU + GPU workers.

Queue layout:
- ``ocr``            : image-download + OCR
- ``cpu_inference``  : text-only scans (runs ViLT on CPU)
- ``gpu_inference``  : multimodal scans (runs fused ViLT on CUDA)
"""
from __future__ import annotations

import os

from celery import Celery

from ots_core.config import get_celery_settings

_s = get_celery_settings()
app = Celery(
    "ots",
    broker=_s.celery_broker_url,
    backend=_s.celery_result_backend,
    include=["backend.worker.tasks.inference_task", "backend.worker.tasks.ocr_task"],
)

app.conf.update(
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_reject_on_worker_lost=True,
    broker_connection_retry_on_startup=True,
    task_routes={
        "ots.scan": {"queue": os.environ.get("DEFAULT_QUEUE", "gpu_inference")},
        "ots.ocr": {"queue": "ocr"},
    },
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=3600,
)
