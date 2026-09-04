"""Benchmark fused kernels vs stock HuggingFace implementation.

Records:
- Kernel launch count (via ``torch.cuda.FX``-level counter if available, else a rough proxy
  using ``torch.profiler``'s ``aten::``-operator count)
- Peak memory (``torch.cuda.max_memory_allocated``)
- P50 / P95 latency from 200 warmed iterations

Run:

    python benchmarks/bench_fusion.py --batch 8 --seq 250 --iters 200

Output: ``benchmarks/results/fusion_<timestamp>.json`` + console table.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class BenchResult:
    label: str
    p50_ms: float
    p95_ms: float
    mean_ms: float
    peak_mem_mb: float
    op_count: int | None


def _profile_op_count(fn) -> int:
    try:
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn()
        return len(prof.events())
    except Exception:
        return -1


def _time_fn(fn, iters: int) -> tuple[float, float, float]:
    times = []
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return (statistics.median(times), times[int(0.95 * len(times)) - 1], statistics.mean(times))


def bench_model(model, inputs, label: str, iters: int) -> BenchResult:
    torch.cuda.reset_peak_memory_stats()

    def run():
        with torch.inference_mode():
            model(**inputs)

    p50, p95, mean = _time_fn(run, iters)
    peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
    ops = _profile_op_count(run)
    return BenchResult(label=label, p50_ms=p50, p95_ms=p95, mean_ms=mean, peak_mem_mb=peak, op_count=ops)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--image-size", type=int, default=384)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--out", type=Path, default=Path("benchmarks/results"))
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available; aborting.")
        return

    from ots_core.fusion import patch_vilt
    from ots_core.models.vilt_multimodal import ViltForOffenseClassification, ViltOffenseConfig

    cfg = ViltOffenseConfig()
    base = ViltForOffenseClassification(cfg).cuda().eval()

    tok = base.processor.tokenizer
    input_ids = torch.randint(10, tok.vocab_size - 10, (args.batch, args.seq), device="cuda")
    attn = torch.ones_like(input_ids)
    pixel_values = torch.randn(args.batch, 3, args.image_size, args.image_size, device="cuda")
    inputs = dict(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attn)

    baseline = bench_model(base, inputs, "baseline", args.iters)

    patch_vilt(base)
    fused = bench_model(base, inputs, "fused", args.iters)

    results = [baseline.__dict__, fused.__dict__]
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"fusion_{int(time.time())}.json"
    path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))

    # Summary
    def _pct(a, b):
        return f"{(1 - b / a) * 100:.1f}%"

    print(f"\nSpeed-up P50: {_pct(baseline.p50_ms, fused.p50_ms)}")
    print(f"Memory reduction: {_pct(baseline.peak_mem_mb, fused.peak_mem_mb)}")
    if baseline.op_count > 0 and fused.op_count > 0:
        print(f"Op-count reduction: {_pct(baseline.op_count, fused.op_count)}")


if __name__ == "__main__":
    main()
