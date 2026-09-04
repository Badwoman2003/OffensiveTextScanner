"""Multimodal offensive-content dataset + modality-aware sampler.

Design points:
- Loads from parquet for fast cold-start; accepts a list of parquet files so mixing sources is
  just concatenation without schema surgery.
- Missing image / text is explicitly encoded via ``modality_mask``; collator produces either a
  zeroed patch tensor or an empty text input so the ViLT model can run unimodally at eval time.
- Random modality dropout (§2.3 of the plan) is applied here, centrally, so every training script
  inherits the same behaviour.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from ots_core.data.schema import ModalityMask


@dataclass
class DataConfig:
    image_size: int = 384
    max_text_len: int = 128
    modality_dropout_p: float = 0.15   # prob of dropping *either* modality per step
    include_ocr_in_text: bool = True
    ocr_sep: str = " [OCR] "


class MultimodalOffenseDataset(Dataset):
    def __init__(
        self,
        parquet_files: Sequence[str | Path],
        config: DataConfig | None = None,
    ) -> None:
        dfs = [pd.read_parquet(p) for p in parquet_files]
        self.df = pd.concat(dfs, ignore_index=True)
        self.cfg = config or DataConfig()

    def __len__(self) -> int:
        return len(self.df)

    def _compose_text(self, text: str, ocr_text: str) -> str:
        if self.cfg.include_ocr_in_text and ocr_text:
            return f"{text}{self.cfg.ocr_sep}{ocr_text}".strip()
        return text.strip()

    def _load_image(self, path: str | None) -> Image.Image | None:
        if not path:
            return None
        try:
            return Image.open(path).convert("RGB").resize(
                (self.cfg.image_size, self.cfg.image_size), Image.BILINEAR
            )
        except Exception:
            return None

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        mask = ModalityMask(row["modality_mask"])
        text = self._compose_text(str(row["text"]), str(row.get("ocr_text") or ""))
        image = self._load_image(row.get("image_path"))

        if random.random() < self.cfg.modality_dropout_p and mask == ModalityMask.BOTH:
            if random.random() < 0.5:
                text = ""
                mask = ModalityMask.IMAGE_ONLY
            else:
                image = None
                mask = ModalityMask.TEXT_ONLY

        return {
            "id": row["id"],
            "text": text,
            "image": image,
            "label": int(row["label"]),
            "source": row["source"],
            "modality_mask": mask.value,
        }


class StratifiedModalitySampler(Sampler[int]):
    """Class-balanced + effective-number weighting over (source, modality, label) strata.

    Reference: Cui et al., "Class-Balanced Loss Based on Effective Number of Samples" (CVPR'19).
    We reuse it as a *sampling* weight here so the training loader sees evenly mixed strata.
    """

    def __init__(self, df: pd.DataFrame, beta: float = 0.999, num_samples: int | None = None) -> None:
        key = df["source"].astype(str) + "|" + df["modality_mask"].astype(str) + "|" + df["label"].astype(str)
        counts = key.value_counts()
        eff_num = {k: (1.0 - beta**c) / (1.0 - beta) for k, c in counts.items()}
        w_per_stratum = {k: 1.0 / v for k, v in eff_num.items()}
        w = key.map(w_per_stratum).astype(float).to_numpy()
        w = w / w.sum()
        self._weights = torch.from_numpy(w)
        self._num_samples = num_samples or len(df)

    def __len__(self) -> int:
        return self._num_samples

    def __iter__(self) -> Iterable[int]:
        g = torch.Generator()
        g.manual_seed(random.randrange(1 << 31))
        idx = torch.multinomial(self._weights, self._num_samples, replacement=True, generator=g)
        return iter(idx.tolist())


def triple_slice(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split a parquet dataframe into text-only / image-only / both slices for eval reports."""
    return {
        "text_only": df[df["modality_mask"] == "text_only"].reset_index(drop=True),
        "image_only": df[df["modality_mask"] == "image_only"].reset_index(drop=True),
        "both": df[df["modality_mask"] == "both"].reset_index(drop=True),
    }


def safe_index(n: int, start: int, end: int) -> range:
    return range(max(0, start), min(n, end))


def sanity_check(ds: MultimodalOffenseDataset, n: int = 8) -> None:
    """Assert that every returned sample is collatable. Used by CI."""
    for i in safe_index(len(ds), 0, n):
        item = ds[i]
        assert "label" in item and isinstance(item["label"], int)
        assert (item["text"] or item["image"] is not None), f"empty sample at {i}"
    print(f"[sanity_check] {min(n, len(ds))} samples OK")


def _ensure_pil(x):
    """Return a PIL.Image if we got a path; pass through otherwise. Used by collators at infer time."""
    if isinstance(x, (str, Path)):
        return Image.open(x).convert("RGB")
    return x


def ceildiv(a: int, b: int) -> int:
    return math.ceil(a / b)
