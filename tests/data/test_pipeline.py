"""Unit tests for data cleaning + stratified sampling. Do not require torch-heavy deps beyond
what the core package already installs."""
from __future__ import annotations

import pandas as pd
import pytest

from ots_core.data.clean.dedup import dedup_texts
from ots_core.data.clean.sanitize import normalise_text
from ots_core.data.dataset import StratifiedModalitySampler


def test_normalise_text_strips_urls_and_mentions():
    raw = "看看这个 http://example.com/foo @某人 #话题# 超级好笑！！！！"
    out = normalise_text(raw)
    assert "http" not in out and "@" not in out
    assert "#" not in out
    assert "话题" in out
    # repeated punctuation capped at two
    assert "！！！" not in out


def test_dedup_texts_drops_near_duplicates():
    df = pd.DataFrame(
        {
            "text": ["今天天气真好呀", "今天天气真好呀!", "毫无关联的另一句"],
            "ocr_text": ["", "", ""],
            "image_path": [None, None, None],
            "label": [0, 0, 0],
            "source": ["coldataset"] * 3,
            "modality_mask": ["text_only"] * 3,
        }
    )
    out = dedup_texts(df)
    assert 2 <= len(out) <= 3  # at minimum the unrelated row survives


def test_stratified_sampler_weights_all_strata():
    df = pd.DataFrame(
        {
            "source": ["coldataset"] * 10 + ["hateful_memes"] * 2,
            "modality_mask": ["text_only"] * 10 + ["both"] * 2,
            "label": [0] * 6 + [1] * 4 + [0, 1],
        }
    )
    s = StratifiedModalitySampler(df, beta=0.9, num_samples=1000)
    counts = {}
    for idx in list(iter(s))[:1000]:
        row = df.iloc[idx]
        counts.setdefault(row["source"], 0)
        counts[row["source"]] += 1
    # minority "hateful_memes" source should be oversampled significantly above its 2/12 prior
    ratio = counts.get("hateful_memes", 0) / 1000
    assert ratio > 0.15, f"expected oversampling, got {ratio:.3f}"
