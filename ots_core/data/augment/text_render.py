"""Synthesise multimodal samples from text-only corpora.

Renders a text-only sample's caption onto a random neutral background, so the model sees the
same text as both *pure text* and *text baked into an image*. Used to mitigate modality
imbalance when most real paired data comes from memes.
"""
from __future__ import annotations

import hashlib
import random
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from ots_core.data.schema import ModalityMask, Sample, Source


def _random_background(size: tuple[int, int], rng: random.Random) -> Image.Image:
    """Solid pastel background — keeps the model from latching onto colour cues as a shortcut."""
    hue = rng.randint(0, 255)
    bg = Image.new("HSV", size, (hue, 40, 240)).convert("RGB")
    return bg


def _load_font(font_path: str | None, size: int) -> ImageFont.ImageFont:
    if font_path and Path(font_path).exists():
        return ImageFont.truetype(font_path, size=size)
    try:
        return ImageFont.truetype("simhei.ttf", size=size)  # Windows Chinese default
    except OSError:
        return ImageFont.load_default()


def render_text_image(text: str, out_path: Path, *, size=(384, 384), font_path: str | None = None) -> Path:
    rng = random.Random(hashlib.sha1(text.encode("utf-8")).digest())
    img = _random_background(size, rng)
    draw = ImageDraw.Draw(img)
    font = _load_font(font_path, size=rng.randint(28, 44))

    chars_per_line = max(6, size[0] // (font.size // 2 + 4))
    wrapped: list[str] = []
    line: list[str] = []
    for ch in text:
        line.append(ch)
        if len(line) >= chars_per_line:
            wrapped.append("".join(line))
            line = []
    if line:
        wrapped.append("".join(line))

    total_h = len(wrapped) * (font.size + 6)
    y = max(8, (size[1] - total_h) // 2)
    for row in wrapped[: (size[1] - 16) // (font.size + 6)]:
        draw.text((16, y), row, fill=(30, 30, 30), font=font)
        y += font.size + 6

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="JPEG", quality=85)
    return out_path


def synthesise_from_text_df(
    df: pd.DataFrame,
    out_dir: Path,
    max_rows: int | None = None,
    font_path: str | None = None,
) -> pd.DataFrame:
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    out_rows: list[dict] = []
    for _, row in df.iterrows():
        if max_rows is not None and len(out_rows) >= max_rows:
            break
        text = str(row["text"])
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
        img_path = img_dir / f"{digest}.jpg"
        if not img_path.exists():
            render_text_image(text, img_path, font_path=font_path)
        sample = Sample(
            id="syn_" + digest,
            text=text,
            ocr_text=text,  # by construction the OCR output equals the original text
            image_path=str(img_path.resolve()),
            label=int(row["label"]),
            source=Source.SYNTHETIC_TEXT_RENDER,
            modality_mask=ModalityMask.BOTH,
        )
        out_rows.append(sample.to_row())
    df_syn = pd.DataFrame(out_rows)
    df_syn.to_parquet(out_dir / "synthetic.parquet", index=False)
    return df_syn
