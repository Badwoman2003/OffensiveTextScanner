"""Lightweight OCR wrapper around RapidOCR (ONNXRuntime, ~10 MB).

Replaces PaddleOCR. Supports both file paths and ``PIL.Image`` inputs, enforces a confidence +
min-char filter, and orders results top-to-bottom / left-to-right so downstream text stays
semantically coherent.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from ots_core.config import get_settings


@dataclass
class OCRLine:
    text: str
    conf: float
    bbox: tuple[float, float, float, float]  # xmin, ymin, xmax, ymax


class RapidOCRExtractor:
    """Wrap RapidOCR with a stable interface; gracefully falls back to a no-op stub if the
    package isn't available (useful for CI where heavy ONNX models aren't installed)."""

    def __init__(self, min_conf: float | None = None, min_chars: int | None = None) -> None:
        settings = get_settings()
        self.min_conf = float(min_conf if min_conf is not None else settings.ocr_min_conf)
        self.min_chars = int(min_chars if min_chars is not None else settings.ocr_min_chars)
        self._engine = self._load_engine()

    @staticmethod
    def _load_engine():
        try:
            from rapidocr_onnxruntime import RapidOCR  # type: ignore
            return RapidOCR()
        except Exception:
            return None

    def _to_numpy(self, image: str | Path | Image.Image | np.ndarray) -> np.ndarray:
        if isinstance(image, np.ndarray):
            return image
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        return np.array(image.convert("RGB"))

    def extract(self, image: str | Path | Image.Image | np.ndarray) -> list[str]:
        lines = self.extract_detailed(image)
        return [ln.text for ln in lines]

    def extract_detailed(self, image) -> list[OCRLine]:
        if self._engine is None:
            return []
        arr = self._to_numpy(image)
        result, _ = self._engine(arr)
        if not result:
            return []

        lines: list[OCRLine] = []
        for box, text, conf in result:
            text = (text or "").strip()
            if conf < self.min_conf or len(text) < self.min_chars:
                continue
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            lines.append(OCRLine(text=text, conf=float(conf), bbox=(min(xs), min(ys), max(xs), max(ys))))

        return self._reading_order(lines)

    @staticmethod
    def _reading_order(lines: Iterable[OCRLine]) -> list[OCRLine]:
        lines = list(lines)
        if not lines:
            return lines
        lines.sort(key=lambda ln: (round(ln.bbox[1] / 20), ln.bbox[0]))
        return lines
