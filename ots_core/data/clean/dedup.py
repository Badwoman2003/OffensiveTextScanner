"""Deduplication utilities.

- Images: perceptual hash (pHash) via ``imagehash``; discard pairs within Hamming <= 4.
- Text: SimHash over character 3-grams for Chinese; discard pairs within Hamming <= 3.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import imagehash
import pandas as pd
from PIL import Image


def _simhash_zh(text: str, bits: int = 64) -> int:
    """SimHash over Chinese character 3-grams. Deterministic, dependency-free."""
    if not text:
        return 0
    grams = [text[i : i + 3] for i in range(max(1, len(text) - 2))]
    v = [0] * bits
    for g in grams:
        h = hash(g) & ((1 << bits) - 1)
        for i in range(bits):
            v[i] += 1 if (h >> i) & 1 else -1
    out = 0
    for i, x in enumerate(v):
        if x > 0:
            out |= 1 << i
    return out


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def dedup_texts(df: pd.DataFrame, threshold: int = 3) -> pd.DataFrame:
    """Drop rows whose text SimHash is within ``threshold`` of an earlier row."""
    keep: list[bool] = []
    buckets: dict[int, list[int]] = defaultdict(list)  # coarse bucket by top-16 bits
    for text in df["text"].fillna(""):
        h = _simhash_zh(str(text))
        bucket = h >> 48
        dup = False
        for prev in buckets[bucket]:
            if _hamming(prev, h) <= threshold:
                dup = True
                break
        if not dup:
            buckets[bucket].append(h)
        keep.append(not dup)
    return df.loc[keep].reset_index(drop=True)


def dedup_images(df: pd.DataFrame, threshold: int = 4) -> pd.DataFrame:
    """Drop rows whose image pHash is within ``threshold`` of an earlier row."""
    keep: list[bool] = []
    seen: list[imagehash.ImageHash] = []
    for path in df["image_path"].fillna(""):
        if not path:
            keep.append(True)
            continue
        p = Path(path)
        if not p.exists():
            keep.append(False)
            continue
        try:
            h = imagehash.phash(Image.open(p).convert("RGB"))
        except Exception:
            keep.append(False)
            continue
        dup = any((h - s) <= threshold for s in seen)
        if not dup:
            seen.append(h)
        keep.append(not dup)
    return df.loc[keep].reset_index(drop=True)


def dedup(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = dedup_texts(df)
    df = dedup_images(df)
    print(f"[dedup] {before} -> {len(df)}")
    return df
