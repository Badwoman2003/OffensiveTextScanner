"""Canonical schema for all ingested samples.

Every ingest / clean / augment step must emit rows that conform to this schema so downstream
`MultimodalOffenseDataset` can treat them uniformly.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Optional

import pyarrow as pa


class ModalityMask(str, Enum):
    TEXT_ONLY = "text_only"
    IMAGE_ONLY = "image_only"
    BOTH = "both"


class Source(str, Enum):
    COLDATASET = "coldataset"
    HATEFUL_MEMES = "hateful_memes"
    TOXICN_MM = "toxicn_mm"
    CHMEME = "chmeme"
    SCRAPED_WEIBO = "scraped_weibo"
    SCRAPED_TIEBA = "scraped_tieba"
    SYNTHETIC_TEXT_RENDER = "synthetic_text_render"


@dataclass
class Sample:
    id: str
    text: str                      # raw caption / post text
    ocr_text: str                  # text extracted from the image (may be empty)
    image_path: Optional[str]      # relative path under data/processed/images; None for text-only
    label: int                     # 0 benign, 1 offensive (later extensible to multi-label)
    source: Source
    modality_mask: ModalityMask

    def to_row(self) -> dict:
        d = asdict(self)
        d["source"] = self.source.value
        d["modality_mask"] = self.modality_mask.value
        return d


ARROW_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("text", pa.string()),
        ("ocr_text", pa.string()),
        ("image_path", pa.string()),
        ("label", pa.int32()),
        ("source", pa.string()),
        ("modality_mask", pa.string()),
    ]
)
