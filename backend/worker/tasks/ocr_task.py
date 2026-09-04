"""Parallel image download + OCR. Can be called directly by ``inference_task`` or as its own
Celery task when we want the OCR stage to live on a dedicated CPU pool."""
from __future__ import annotations

import asyncio
import io
import logging
from typing import Iterable

import httpx
from PIL import Image

from backend.worker.celery_app import app
from ots_core.ocr.rapid_ocr import RapidOCRExtractor

log = logging.getLogger("ots.ocr")
_ocr: RapidOCRExtractor | None = None


def _get_ocr() -> RapidOCRExtractor:
    global _ocr
    if _ocr is None:
        _ocr = RapidOCRExtractor()
    return _ocr


async def _download(client: httpx.AsyncClient, url: str) -> Image.Image | None:
    try:
        r = await client.get(url, timeout=10.0, follow_redirects=True)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception as e:
        log.warning("failed to download %s: %s", url, e)
        return None


async def _fetch_all(urls: Iterable[str]) -> list[Image.Image | None]:
    async with httpx.AsyncClient(headers={"User-Agent": "OTS/0.2"}) as client:
        return await asyncio.gather(*(_download(client, u) for u in urls))


def run_ocr(image_urls: list[str]) -> list[tuple[str, str]]:
    """Returns [(url, ocr_text), ...]; failed downloads produce (url, "")."""
    images = asyncio.run(_fetch_all(image_urls))
    extractor = _get_ocr()
    out: list[tuple[str, str]] = []
    for url, img in zip(image_urls, images):
        if img is None:
            out.append((url, ""))
            continue
        try:
            lines = extractor.extract(img)
            out.append((url, " ".join(lines)))
        except Exception as e:
            log.warning("ocr failed on %s: %s", url, e)
            out.append((url, ""))
    return out


@app.task(name="ots.ocr", queue="ocr")
def ocr_task(image_urls: list[str]) -> list[tuple[str, str]]:
    return run_ocr(image_urls)
