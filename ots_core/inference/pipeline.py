"""End-to-end scanner pipeline used by the worker.

Encapsulates:
- OCR on raw images
- Tokenisation + image preprocessing via ``ViltProcessor``
- ViLT forward pass (with fused kernels if available)
- Post-processing into ``ScanResult`` rows for the API response
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

from ots_core.config import get_settings
from ots_core.fusion import patch_vilt, triton_available
from ots_core.models.vilt_multimodal import ViltForOffenseClassification, ViltOffenseConfig
from ots_core.ocr.rapid_ocr import RapidOCRExtractor


@dataclass
class ScanItem:
    text: str = ""
    image: Optional[Image.Image | str | Path] = None


@dataclass
class ScanResult:
    text: str
    ocr_text: str
    label: int
    prob_offensive: float
    modality_used: str


class ScannerPipeline:
    """Thread-safe-ish inference wrapper — instantiate once per worker process."""

    def __init__(self, checkpoint_dir: str | Path | None = None) -> None:
        settings = get_settings()
        self.device = torch.device(settings.device if torch.cuda.is_available() else "cpu")
        base_cfg = ViltOffenseConfig()
        ckpt = Path(checkpoint_dir or settings.model_dir)
        if ckpt.exists() and (ckpt / "pytorch_model.bin").exists():
            self.model = ViltForOffenseClassification.load(ckpt, base_cfg)
        else:
            self.model = ViltForOffenseClassification(base_cfg)
        self.model.to(self.device).eval()

        if settings.use_fused_kernels and triton_available() and self.device.type == "cuda":
            patch_vilt(self.model)

        self.processor = self.model.processor
        self.max_len = settings.max_text_len
        self.image_size = settings.image_size
        self.ocr = RapidOCRExtractor()

    @staticmethod
    def _open_image(image) -> Optional[Image.Image]:
        if image is None:
            return None
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        return Image.open(image).convert("RGB")

    def _maybe_ocr(self, image: Optional[Image.Image]) -> str:
        if image is None:
            return ""
        try:
            return " ".join(self.ocr.extract(image))
        except Exception:
            return ""

    def _compose(self, text: str, ocr_text: str) -> str:
        if ocr_text:
            return f"{text} [OCR] {ocr_text}".strip()
        return text or "[EMPTY]"

    @torch.inference_mode()
    def batch_scan(self, items: list[ScanItem]) -> list[ScanResult]:
        images: list[Image.Image] = []
        texts: list[str] = []
        ocr_texts: list[str] = []
        modality: list[str] = []

        blank = Image.new("RGB", (self.image_size, self.image_size), (255, 255, 255))
        for it in items:
            img = self._open_image(it.image)
            ocr_text = self._maybe_ocr(img)
            ocr_texts.append(ocr_text)
            images.append(img or blank)
            if img is not None and it.text:
                modality.append("both")
            elif img is not None:
                modality.append("image_only")
            else:
                modality.append("text_only")
            texts.append(self._compose(it.text, ocr_text))

        enc = self.processor(
            images=images, text=texts,
            padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt",
        )
        enc = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in enc.items()}
        out = self.model(
            pixel_values=enc["pixel_values"],
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            token_type_ids=enc.get("token_type_ids"),
            pixel_mask=enc.get("pixel_mask"),
        )
        probs = torch.softmax(out["logits"], dim=-1)
        labels = probs.argmax(dim=-1).tolist()
        p_off = probs[:, -1].tolist()

        results: list[ScanResult] = []
        for it, ocr_text, lbl, p, mm in zip(items, ocr_texts, labels, p_off, modality):
            results.append(
                ScanResult(
                    text=it.text,
                    ocr_text=ocr_text,
                    label=int(lbl),
                    prob_offensive=float(p),
                    modality_used=mm,
                )
            )
        return results
