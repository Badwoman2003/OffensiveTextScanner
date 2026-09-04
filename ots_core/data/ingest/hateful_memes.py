"""Ingest Facebook Hateful Memes + machine-translate the English captions to Chinese.

Expected layout (from the challenge release):

    data/raw/hateful_memes/
        train.jsonl, dev_seen.jsonl, test_seen.jsonl
        img/*.png

Each line: ``{"id": 42953, "img": "img/42953.png", "label": 1, "text": "..."}``.

Translation is pluggable via ``--translator``. By default we use ``NoopTranslator`` so tests
don't depend on the network; swap in ``OpenAITranslator`` / ``HfTranslator`` in production.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator, Protocol

import pandas as pd

from ots_core.data.schema import ModalityMask, Sample, Source


class Translator(Protocol):
    def translate(self, text: str) -> str: ...


class NoopTranslator:
    def translate(self, text: str) -> str:  # pragma: no cover - trivial
        return text


class HfTranslator:
    """Batch-translate EN->ZH using a HuggingFace seq2seq (e.g. Helsinki-NLP/opus-mt-en-zh)."""

    def __init__(self, model_name: str = "Helsinki-NLP/opus-mt-en-zh", device: str = "cpu") -> None:
        from transformers import pipeline  # lazy import

        self._pipe = pipeline("translation", model=model_name, device=-1 if device == "cpu" else 0)

    def translate(self, text: str) -> str:
        if not text.strip():
            return text
        out = self._pipe(text, max_length=256, truncation=True)
        return out[0]["translation_text"]


def iter_hateful_memes(root: Path, split: str, translator: Translator) -> Iterator[Sample]:
    jsonl = root / f"{split}.jsonl"
    if not jsonl.exists():
        raise FileNotFoundError(f"Hateful Memes split missing: {jsonl}")
    with jsonl.open(encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text_en: str = obj.get("text", "").strip()
            text_zh = translator.translate(text_en)
            img_rel = obj["img"]
            img_path = root / img_rel
            if not img_path.exists():
                continue
            yield Sample(
                id=f"hm_{obj['id']}",
                text=text_zh,
                ocr_text=text_en,
                image_path=str(img_path.resolve()),
                label=int(obj.get("label", 0)),
                source=Source.HATEFUL_MEMES,
                modality_mask=ModalityMask.BOTH,
            )


def ingest(root: Path, out_dir: Path, translator: Translator | None = None) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tr = translator or NoopTranslator()
    counts: dict[str, int] = {}
    for split in ("train", "dev_seen", "test_seen"):
        if not (root / f"{split}.jsonl").exists():
            continue
        rows = [s.to_row() for s in iter_hateful_memes(root, split, tr)]
        pd.DataFrame(rows).to_parquet(out_dir / f"hateful_memes_{split}.parquet", index=False)
        counts[split] = len(rows)
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/raw/hateful_memes"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/hateful_memes"))
    ap.add_argument("--translator", choices=["noop", "hf"], default="noop")
    args = ap.parse_args()
    tr: Translator = NoopTranslator() if args.translator == "noop" else HfTranslator()
    counts = ingest(args.root, args.out, tr)
    print(f"[hateful_memes] ingested counts: {counts}")


if __name__ == "__main__":
    main()
