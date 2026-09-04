"""Ingest the COLDataset (Chinese Offensive Language).

Reference: Deng et al. 2022, "COLD: A Benchmark for Chinese Offensive Language Detection".
Expected on-disk layout (downloaded manually from GitHub):

    data/raw/COLDataset/
        COLDataset/train.csv
        COLDataset/dev.csv
        COLDataset/test.csv

Each CSV has columns: `TEXT,label,topic,fine-grained-label`.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Iterator

import pandas as pd

from ots_core.data.schema import ModalityMask, Sample, Source


def _stable_id(text: str) -> str:
    return "col_" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def iter_coldataset(root: Path, split: str) -> Iterator[Sample]:
    csv = root / "COLDataset" / f"{split}.csv"
    if not csv.exists():
        raise FileNotFoundError(f"COLDataset file missing: {csv}. Download it first.")
    df = pd.read_csv(csv)
    text_col = "TEXT" if "TEXT" in df.columns else df.columns[0]
    label_col = "label" if "label" in df.columns else df.columns[1]
    for _, row in df.iterrows():
        text = str(row[text_col]).strip()
        if not text:
            continue
        try:
            label = int(row[label_col])
        except (TypeError, ValueError):
            continue
        yield Sample(
            id=_stable_id(text),
            text=text,
            ocr_text="",
            image_path=None,
            label=int(bool(label)),
            source=Source.COLDATASET,
            modality_mask=ModalityMask.TEXT_ONLY,
        )


def ingest(root: Path, out_dir: Path) -> dict[str, int]:
    """Dump COLDataset splits as parquet under ``out_dir``. Returns per-split counts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for split in ("train", "dev", "test"):
        rows = [s.to_row() for s in iter_coldataset(root, split)]
        pd.DataFrame(rows).to_parquet(out_dir / f"coldataset_{split}.parquet", index=False)
        counts[split] = len(rows)
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/raw/COLDataset"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/coldataset"))
    args = ap.parse_args()
    counts = ingest(args.root, args.out)
    print(f"[coldataset] ingested counts: {counts}")


if __name__ == "__main__":
    main()
