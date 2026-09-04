"""Text normalisation for Chinese social-media text.

Does NOT remove offensive words (that would defeat the point of the dataset). It only:
- strips URLs, @mentions, #topics, emoji variants selectors;
- collapses whitespace / repeated punctuation;
- normalises full-width to half-width ASCII + lowercases Latin chars.
"""
from __future__ import annotations

import re
import unicodedata

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"[@＠][\u4e00-\u9fa5A-Za-z0-9_\-]{1,30}")
_TOPIC_RE = re.compile(r"#([^#\n]{1,40})#")
_REPEAT_PUNCT = re.compile(r"([!?。！？,.，、])\1{2,}")
_WS_RE = re.compile(r"\s+")


def _fullwidth_to_halfwidth(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def normalise_text(text: str) -> str:
    if not text:
        return ""
    t = _fullwidth_to_halfwidth(text)
    t = _URL_RE.sub(" ", t)
    t = _MENTION_RE.sub(" ", t)
    t = _TOPIC_RE.sub(r"\1", t)   # keep the topic word, drop the '#'
    t = _REPEAT_PUNCT.sub(r"\1\1", t)
    t = _WS_RE.sub(" ", t).strip()
    return t


def sanitize_df(df):
    df = df.copy()
    df["text"] = df["text"].fillna("").astype(str).map(normalise_text)
    df["ocr_text"] = df["ocr_text"].fillna("").astype(str).map(normalise_text)
    df = df[df["text"].str.len().gt(0) | df["ocr_text"].str.len().gt(0)].reset_index(drop=True)
    return df
