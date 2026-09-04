"""Export a trained OTS ViLT checkpoint to ONNX for cross-runtime deployment (Triton Inference
Server, ONNX Runtime, etc.).

Note: fused Triton kernels are PyTorch-only; the ONNX export uses the *unfused* forward path
so the resulting graph is portable. Use ``--fuse`` to additionally emit a TorchScript bundle
that keeps the fused kernels in the ``forward``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ots_core.fusion import patch_vilt
from ots_core.models.vilt_multimodal import ViltForOffenseClassification, ViltOffenseConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fuse", action="store_true")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--image-size", type=int, default=384)
    args = ap.parse_args()

    model = ViltForOffenseClassification.load(args.ckpt, ViltOffenseConfig())
    model.eval()
    if args.fuse and torch.cuda.is_available():
        model.cuda()
        patch_vilt(model)

    dummy = {
        "pixel_values": torch.randn(args.batch, 3, args.image_size, args.image_size),
        "input_ids": torch.randint(0, 20000, (args.batch, args.seq), dtype=torch.long),
        "attention_mask": torch.ones(args.batch, args.seq, dtype=torch.long),
    }

    if args.fuse:
        out_pt = args.out.with_suffix(".pt")
        scripted = torch.jit.trace(model, (dummy["pixel_values"], dummy["input_ids"], dummy["attention_mask"]))
        scripted.save(str(out_pt))
        print(f"[export] fused TorchScript -> {out_pt}")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(
            model,
            (dummy["pixel_values"], dummy["input_ids"], dummy["attention_mask"]),
            str(args.out),
            opset_version=args.opset,
            input_names=["pixel_values", "input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "pixel_values": {0: "batch"},
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "logits": {0: "batch"},
            },
        )
        print(f"[export] ONNX -> {args.out}")


if __name__ == "__main__":
    main()
