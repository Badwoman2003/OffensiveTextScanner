"""Stage-B: supervised fine-tuning on the mixed multimodal dataset.

Key ingredients (all plumbed through cleanly so Stage-C just inherits + swaps the loss):
- ``MultimodalOffenseDataset`` with modality dropout
- ``StratifiedModalitySampler`` — class-balanced weighting by (source, modality, label)
- Focal loss (γ=2) by default; BCE fallback via ``--loss ce``
- Three-slice evaluation (text_only / image_only / both) every epoch

CLI::

    python -m ots_core.training.stage_b_finetune \
        --train data/processed/train_mixed.parquet \
        --val   data/processed/val_mixed.parquet \
        --init  checkpoints/vilt-zh-cpt/final \
        --out   checkpoints/vilt-offense-stageB
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from ots_core.data.collator import ViltCollator
from ots_core.data.dataset import DataConfig, MultimodalOffenseDataset, StratifiedModalitySampler
from ots_core.data.evaluate import evaluate_all
from ots_core.models.losses import FocalLoss
from ots_core.models.vilt_multimodal import ViltForOffenseClassification, ViltOffenseConfig


@dataclass
class FineTuneConfig:
    train_parquet: Path
    val_parquet: Path
    init_dir: Path | None
    out_dir: Path
    epochs: int = 4
    batch_size: int = 32
    lr: float = 3e-5
    warmup_ratio: float = 0.06
    modality_dropout_p: float = 0.15
    loss: Literal["focal", "ce"] = "focal"
    focal_gamma: float = 2.0
    grad_accum: int = 1
    fp16: bool = True
    num_workers: int = 4
    seed: int = 42


def build_model(cfg: FineTuneConfig, device: torch.device) -> ViltForOffenseClassification:
    base_cfg = ViltOffenseConfig()
    if cfg.init_dir is not None and (cfg.init_dir / "pytorch_model.bin").exists():
        model = ViltForOffenseClassification.load(cfg.init_dir, base_cfg)
    else:
        model = ViltForOffenseClassification(base_cfg)
    return model.to(device)


def build_loaders(cfg: FineTuneConfig, processor) -> tuple[DataLoader, DataLoader]:
    data_cfg = DataConfig(modality_dropout_p=cfg.modality_dropout_p)
    train_ds = MultimodalOffenseDataset([cfg.train_parquet], config=data_cfg)
    val_ds = MultimodalOffenseDataset([cfg.val_parquet], config=DataConfig(modality_dropout_p=0.0))
    sampler = StratifiedModalitySampler(train_ds.df)
    collator = ViltCollator(processor=processor)
    return (
        DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler, collate_fn=collator,
                   num_workers=cfg.num_workers, pin_memory=True),
        DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collator,
                   num_workers=cfg.num_workers, pin_memory=True),
    )


def _loss_fn(cfg: FineTuneConfig) -> torch.nn.Module:
    if cfg.loss == "focal":
        return FocalLoss(gamma=cfg.focal_gamma)
    return torch.nn.CrossEntropyLoss()


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    y_true, y_pred, mm, srcs = [], [], [], []
    for batch in tqdm(loader, desc="eval", leave=False):
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        out = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attn)
        pred = out["logits"].argmax(dim=-1).cpu().tolist()
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(pred)
        mm.extend(batch["modality_mask"])
    return evaluate_all(y_true, y_pred, mm, srcs or None)


def train(cfg: FineTuneConfig) -> dict:
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, device)
    train_loader, val_loader = build_loaders(cfg, model.processor)
    loss_fn = _loss_fn(cfg)

    total_steps = (len(train_loader) // cfg.grad_accum) * cfg.epochs
    warmup_steps = int(cfg.warmup_ratio * total_steps)
    optim = AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim,
        lr_lambda=lambda s: min((s + 1) / max(1, warmup_steps), 1.0)
        * max(0.0, (total_steps - s) / max(1, total_steps - warmup_steps)),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.fp16 and device.type == "cuda")

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_f1 = -1.0
    step = 0

    for epoch in range(cfg.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"ft-epoch-{epoch}")
        for batch in pbar:
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attn = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                out = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attn)
                loss = loss_fn(out["logits"], labels) / cfg.grad_accum

            scaler.scale(loss).backward()
            if (step + 1) % cfg.grad_accum == 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
                sched.step()
                optim.zero_grad(set_to_none=True)
            pbar.set_postfix(loss=float(loss))
            step += 1

        report = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, **report})
        print(json.dumps(report, ensure_ascii=False, indent=2))

        if report["overall"]["f1_macro"] > best_f1:
            best_f1 = report["overall"]["f1_macro"]
            model.save(cfg.out_dir / "best")
        model.save(cfg.out_dir / f"epoch-{epoch}")

    (cfg.out_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"best_f1_macro": best_f1, "history": history}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--init", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--loss", choices=["focal", "ce"], default="focal")
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    cfg = FineTuneConfig(
        train_parquet=args.train, val_parquet=args.val, init_dir=args.init, out_dir=args.out,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, loss=args.loss, fp16=args.fp16,
    )
    train(cfg)


if __name__ == "__main__":
    main()
