"""Stage-A: Chinese continued pretraining (CPT) for ViLT.

Objectives (exactly as in the original ViLT paper, retargeted to Chinese image-caption pairs):
  - MLM    (Masked Language Modelling) on the text tokens (prob 15%).
  - ITM    (Image-Text Matching) using in-batch negatives (50% swap).
  - WPA    (Word-Patch Alignment) via optimal-transport IPOT on last-layer representations.

This script is self-contained but lightweight: it handles data plumbing, logging, checkpointing,
and mixed precision. At call site::

    python -m ots_core.training.stage_a_cpt \
        --pairs data/processed/cc3m_zh.parquet \
        --out checkpoints/vilt-zh-cpt \
        --batch-size 64 --epochs 3
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ots_core.models.vilt_multimodal import (
    ViltForOffenseClassification,
    ViltOffenseConfig,
)
from ots_core.training.wpa import WordPatchAlignment

MLM_PROB = 0.15


@dataclass
class CPTConfig:
    pairs_parquet: Path = Path("data/processed/cc3m_zh.parquet")
    out_dir: Path = Path("checkpoints/vilt-zh-cpt")
    epochs: int = 3
    batch_size: int = 64
    lr: float = 1e-4
    warmup_ratio: float = 0.05
    max_text_len: int = 40
    image_size: int = 384
    mlm_weight: float = 1.0
    itm_weight: float = 1.0
    wpa_weight: float = 0.1
    grad_accum: int = 1
    fp16: bool = True
    seed: int = 42
    log_every: int = 50
    save_every: int = 1000


class ImageTextPairs(Dataset):
    def __init__(self, parquet: Path, image_size: int) -> None:
        self.df = pd.read_parquet(parquet)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        try:
            img = Image.open(row["image_path"]).convert("RGB").resize(
                (self.image_size, self.image_size), Image.BILINEAR
            )
        except Exception:
            img = Image.new("RGB", (self.image_size, self.image_size), (255, 255, 255))
        return {"image": img, "text": str(row.get("text") or ""), "idx": idx}


def _mlm_mask(
    input_ids: torch.Tensor, tokenizer, prob: float = MLM_PROB
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (masked_input_ids, labels_for_mlm) — labels are -100 where we keep originals."""
    labels = input_ids.clone()
    mask = (torch.rand(input_ids.shape, device=input_ids.device) < prob)
    specials = torch.tensor(
        tokenizer.get_special_tokens_mask(input_ids.tolist()[0], already_has_special_tokens=True),
        device=input_ids.device,
        dtype=torch.bool,
    ) if input_ids.dim() == 2 and input_ids.size(0) == 1 else None
    if specials is not None:
        mask = mask & ~specials
    mask = mask & (input_ids != tokenizer.pad_token_id)

    masked = input_ids.clone()
    masked[mask] = tokenizer.mask_token_id
    labels[~mask] = -100
    return masked, labels


def _swap_for_itm(batch_imgs: torch.Tensor, swap_ratio: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
    """Shuffle half the batch so (image_i, text_i) becomes negative. Returns new pixel_values + label."""
    n = batch_imgs.size(0)
    labels = torch.ones(n, dtype=torch.long, device=batch_imgs.device)
    swapped = batch_imgs.clone()
    perm = torch.randperm(n, device=batch_imgs.device)
    swap_mask = torch.rand(n, device=batch_imgs.device) < swap_ratio
    idx = torch.where(swap_mask, perm, torch.arange(n, device=batch_imgs.device))
    swapped = batch_imgs[idx]
    labels[swap_mask] = 0
    return swapped, labels


def train_cpt(cfg: CPTConfig) -> None:
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = ViltOffenseConfig()
    model = ViltForOffenseClassification(model_cfg).to(device)
    processor = model.processor
    tokenizer = processor.tokenizer

    hidden = model.config.hidden_size
    mlm_head = nn.Linear(hidden, tokenizer.vocab_size).to(device)
    itm_head = nn.Linear(hidden, 2).to(device)
    wpa = WordPatchAlignment().to(device)

    ds = ImageTextPairs(cfg.pairs_parquet, cfg.image_size)
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=lambda batch: _cpt_collate(batch, processor, cfg.max_text_len),
        pin_memory=True,
    )

    params = list(model.parameters()) + list(mlm_head.parameters()) + list(itm_head.parameters())
    optim = AdamW(params, lr=cfg.lr, betas=(0.9, 0.98), weight_decay=0.01)
    total = math.ceil(len(loader) / cfg.grad_accum) * cfg.epochs
    warmup = int(total * cfg.warmup_ratio)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim,
        lr_lambda=lambda step: min((step + 1) / max(1, warmup), 1.0)
        * max(0.0, (total - step) / max(1, total - warmup)),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.fp16 and device.type == "cuda")

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    step = 0
    for epoch in range(cfg.epochs):
        model.train()
        pbar = tqdm(loader, desc=f"cpt-epoch-{epoch}")
        for batch in pbar:
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attn = batch["attention_mask"].to(device, non_blocking=True)

            mlm_ids, mlm_labels = _mlm_mask(input_ids, tokenizer)
            itm_pixels, itm_labels = _swap_for_itm(pixel_values)

            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                out = model(
                    pixel_values=itm_pixels,
                    input_ids=mlm_ids,
                    attention_mask=attn,
                )
                last = out["last_hidden_state"]  # (B, Ltxt + Limg, H)
                Ltxt = mlm_ids.size(1)
                text_hidden = last[:, :Ltxt, :]
                image_hidden = last[:, Ltxt:, :]
                cls_hidden = last[:, 0, :]

                mlm_logits = mlm_head(text_hidden)
                itm_logits = itm_head(cls_hidden)

                loss_mlm = nn.functional.cross_entropy(
                    mlm_logits.reshape(-1, mlm_logits.size(-1)),
                    mlm_labels.reshape(-1),
                    ignore_index=-100,
                )
                loss_itm = nn.functional.cross_entropy(itm_logits, itm_labels)
                loss_wpa = wpa(text_hidden, image_hidden, attn)

                loss = (
                    cfg.mlm_weight * loss_mlm
                    + cfg.itm_weight * loss_itm
                    + cfg.wpa_weight * loss_wpa
                ) / cfg.grad_accum

            scaler.scale(loss).backward()
            if (step + 1) % cfg.grad_accum == 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(optim)
                scaler.update()
                sched.step()
                optim.zero_grad(set_to_none=True)

            if step % cfg.log_every == 0:
                pbar.set_postfix(mlm=float(loss_mlm), itm=float(loss_itm), wpa=float(loss_wpa))
            if step and step % cfg.save_every == 0:
                model.save(cfg.out_dir / f"step-{step}")
            step += 1

        model.save(cfg.out_dir / f"epoch-{epoch}")

    model.save(cfg.out_dir / "final")
    (cfg.out_dir / "cpt_config.json").write_text(
        json.dumps({k: str(v) for k, v in cfg.__dict__.items()}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _cpt_collate(batch, processor, max_len):
    images = [b["image"] for b in batch]
    texts = [b["text"] or "[UNK]" for b in batch]
    enc = processor(
        images=images, text=texts, padding="max_length",
        truncation=True, max_length=max_len, return_tensors="pt",
    )
    return enc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    cfg = CPTConfig(
        pairs_parquet=args.pairs, out_dir=args.out,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, fp16=args.fp16,
    )
    train_cpt(cfg)


if __name__ == "__main__":
    main()
