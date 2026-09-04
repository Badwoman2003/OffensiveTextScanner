"""Collator that turns ``MultimodalOffenseDataset`` items into ViLT-ready batches."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image
from transformers import ViltProcessor


@dataclass
class ViltCollator:
    processor: ViltProcessor
    max_text_len: int = 128
    image_size: int = 384

    def _blank_image(self) -> Image.Image:
        return Image.new("RGB", (self.image_size, self.image_size), (255, 255, 255))

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        texts: list[str] = []
        images: list[Image.Image] = []
        labels: list[int] = []
        modality_mask: list[str] = []

        for item in batch:
            texts.append(item.get("text") or "[EMPTY]")
            images.append(item.get("image") or self._blank_image())
            labels.append(int(item["label"]))
            modality_mask.append(item["modality_mask"])

        enc = self.processor(
            images=images,
            text=texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt",
        )
        enc["labels"] = torch.tensor(labels, dtype=torch.long)
        enc["modality_mask"] = modality_mask  # passed through for Gradient-Blending weighting
        return enc
