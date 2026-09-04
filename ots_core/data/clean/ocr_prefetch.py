"""Pre-run OCR over all multimodal rows so train-time ``ocr_text`` matches inference-time OCR.

Uses the same RapidOCR wrapper as production inference, guaranteeing distribution parity.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ots_core.ocr.rapid_ocr import RapidOCRExtractor


def prefetch_ocr(df: pd.DataFrame, extractor: RapidOCRExtractor | None = None) -> pd.DataFrame:
    extractor = extractor or RapidOCRExtractor()
    df = df.copy()
    for i, row in tqdm(df.iterrows(), total=len(df), desc="ocr"):
        path = row.get("image_path")
        if not path:
            continue
        try:
            lines = extractor.extract(path)
        except Exception:
            lines = []
        df.at[i, "ocr_text"] = " ".join(lines)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True, help="input parquet with image_path")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.input)
    df = prefetch_ocr(df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"[ocr_prefetch] {len(df)} rows -> {args.output}")


if __name__ == "__main__":
    main()
