"""Stage-C: modality-imbalance specialist training.

On top of the Stage-B checkpoint, we:
1. Attach two auxiliary ``ModalityHead``s (text-only, image-only) to the backbone.
2. Feed every batch through the backbone three times (text_only view, image_only view, both).
3. Optimise ``GradientBlendedLoss`` whose weights are recomputed after every epoch via
   ``solve_weights`` using the latest train/val statistics.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from ots_core.data.collator import ViltCollator
from ots_core.data.dataset import DataConfig, MultimodalOffenseDataset, StratifiedModalitySampler
from ots_core.data.evaluate import evaluate_all
from ots_core.models.heads import ModalityHead
from ots_core.models.losses import FocalLoss, GradientBlendedLoss
from ots_core.models.vilt_multimodal import ViltForOffenseClassification, ViltOffenseConfig
from ots_core.training.gradient_blending import StreamStats, solve_weights


@dataclass
class StageCConfig:
    train_parquet: Path
    val_parquet: Path
    init_dir: Path
    out_dir: Path
    epochs: int = 3
    batch_size: int = 16
    lr: float = 1e-5
    seed: int = 42


class _ModalityAwareViltWrapper(torch.nn.Module):
    """Wrap the base model and expose text-only / image-only / multi outputs."""

    def __init__(self, base: ViltForOffenseClassification) -> None:
        super().__init__()
        self.base = base
        hidden = base.config.hidden_size
        self.head_text = ModalityHead(hidden, base.config.num_labels)
        self.head_image = ModalityHead(hidden, base.config.num_labels)

    def forward(self, pixel_values, input_ids, attention_mask) -> dict:
        B = pixel_values.size(0)
        blank_ids = torch.full_like(input_ids, fill_value=self.base.processor.tokenizer.pad_token_id)
        blank_attn = torch.zeros_like(attention_mask)
        blank_pix = torch.zeros_like(pixel_values)

        out_txt = self.base(pixel_values=blank_pix, input_ids=input_ids, attention_mask=attention_mask)
        out_img = self.base(pixel_values=pixel_values, input_ids=blank_ids, attention_mask=blank_attn)
        out_mm = self.base(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)

        return {
            "logits_text": self.head_text(out_txt["pooler_output"]),
            "logits_image": self.head_image(out_img["pooler_output"]),
            "logits_multi": out_mm["logits"],
        }


def run(cfg: StageCConfig) -> None:
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base = ViltForOffenseClassification.load(cfg.init_dir, ViltOffenseConfig())
    model = _ModalityAwareViltWrapper(base).to(device)

    data_cfg = DataConfig(modality_dropout_p=0.0)
    train_ds = MultimodalOffenseDataset([cfg.train_parquet], data_cfg)
    val_ds = MultimodalOffenseDataset([cfg.val_parquet], data_cfg)
    sampler = StratifiedModalitySampler(train_ds.df)
    collator = ViltCollator(processor=base.processor)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler, collate_fn=collator, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collator, num_workers=4)

    gb = GradientBlendedLoss(FocalLoss(gamma=2.0))
    optim = AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)

    prev_stats = {"text": (1.0, 1.0), "image": (1.0, 1.0), "multi": (1.0, 1.0)}  # (train, val)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.epochs):
        model.train()
        ep_train = {"text": 0.0, "image": 0.0, "multi": 0.0, "n": 0}
        for batch in tqdm(train_loader, desc=f"stageC-{epoch}"):
            pix = batch["pixel_values"].to(device)
            ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)

            out = model(pix, ids, attn)
            parts = gb(out["logits_text"], out["logits_image"], out["logits_multi"], y)
            optim.zero_grad()
            parts["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            ep_train["text"] += float(parts["loss_text"]) * y.size(0)
            ep_train["image"] += float(parts["loss_image"]) * y.size(0)
            ep_train["multi"] += float(parts["loss_multi"]) * y.size(0)
            ep_train["n"] += y.size(0)

        ep_val = _eval_streams(model, val_loader, gb.base, device)
        stats = {
            name: StreamStats(
                train_loss_prev=prev_stats[name][0], train_loss_now=ep_train[name] / max(1, ep_train["n"]),
                val_loss_prev=prev_stats[name][1], val_loss_now=ep_val[name],
            )
            for name in ("text", "image", "multi")
        }
        w_t, w_i, w_m = solve_weights(stats["text"], stats["image"], stats["multi"])
        gb.update_weights(w_t, w_i, w_m)
        prev_stats = {
            "text": (ep_train["text"] / max(1, ep_train["n"]), ep_val["text"]),
            "image": (ep_train["image"] / max(1, ep_train["n"]), ep_val["image"]),
            "multi": (ep_train["multi"] / max(1, ep_train["n"]), ep_val["multi"]),
        }
        (cfg.out_dir / f"gb_weights_epoch_{epoch}.json").write_text(
            json.dumps({"w_text": w_t, "w_image": w_i, "w_multi": w_m}, indent=2), encoding="utf-8"
        )
        base.save(cfg.out_dir / f"epoch-{epoch}")

    base.save(cfg.out_dir / "final")


@torch.no_grad()
def _eval_streams(model, loader, loss_fn, device) -> dict[str, float]:
    model.eval()
    tot = {"text": 0.0, "image": 0.0, "multi": 0.0, "n": 0}
    y_true, y_pred, mm = [], [], []
    for batch in loader:
        pix = batch["pixel_values"].to(device)
        ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        y = batch["labels"].to(device)
        out = model(pix, ids, attn)
        for key in ("text", "image", "multi"):
            tot[key] += float(loss_fn(out[f"logits_{key}"], y)) * y.size(0)
        tot["n"] += y.size(0)
        y_true.extend(y.cpu().tolist())
        y_pred.extend(out["logits_multi"].argmax(-1).cpu().tolist())
        mm.extend(batch["modality_mask"])
    print(json.dumps(evaluate_all(y_true, y_pred, mm), ensure_ascii=False, indent=2))
    return {k: tot[k] / max(1, tot["n"]) for k in ("text", "image", "multi")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--init", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()
    run(StageCConfig(args.train, args.val, args.init, args.out, args.epochs))


if __name__ == "__main__":
    main()
