"""Modality-aware evaluation: reports F1 on text_only / image_only / both slices
plus macro-F1 and the "modality robustness gap" = max slice F1 − min slice F1.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import classification_report, f1_score


@dataclass
class SliceReport:
    slice_name: str
    n: int
    f1_macro: float
    f1_pos: float
    per_class: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "slice": self.slice_name,
            "n": self.n,
            "f1_macro": round(self.f1_macro, 4),
            "f1_offensive": round(self.f1_pos, 4),
            "per_class": self.per_class,
        }


def evaluate_slice(y_true, y_pred, name: str) -> SliceReport:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return SliceReport(slice_name=name, n=0, f1_macro=float("nan"), f1_pos=float("nan"))
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_pos = f1_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)
    per_class = classification_report(
        y_true, y_pred, target_names=["benign", "offensive"], output_dict=True, zero_division=0
    )
    return SliceReport(name, len(y_true), float(f1_macro), float(f1_pos), per_class)


def evaluate_all(
    y_true, y_pred, modality_mask, sources=None
) -> dict[str, Any]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mm = np.asarray(modality_mask)

    slices = {}
    for key in ("text_only", "image_only", "both"):
        idx = (mm == key)
        slices[key] = evaluate_slice(y_true[idx], y_pred[idx], key).to_dict()

    overall = evaluate_slice(y_true, y_pred, "overall").to_dict()

    valid = [slices[k]["f1_macro"] for k in slices if slices[k]["n"] > 0]
    gap = (max(valid) - min(valid)) if valid else float("nan")

    by_source = {}
    if sources is not None:
        sources = np.asarray(sources)
        for src in sorted(set(sources.tolist())):
            idx = sources == src
            by_source[src] = evaluate_slice(y_true[idx], y_pred[idx], f"src:{src}").to_dict()

    return {
        "overall": overall,
        "slices": slices,
        "modality_robustness_gap": round(gap, 4),
        "by_source": by_source,
    }
