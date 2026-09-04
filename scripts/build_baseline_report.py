"""Aggregate training histories into a single Markdown baseline comparison table.

Reads JSON histories produced by Stage-B / Stage-C runs (and optionally the legacy BERT baseline)
then emits ``docs/baseline_report.md`` with per-slice F1 comparison.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _row(name: str, rep: dict) -> str:
    sl = rep.get("slices", {})
    return (
        f"| {name} "
        f"| {rep.get('overall', {}).get('f1_macro', '-')} "
        f"| {sl.get('text_only', {}).get('f1_macro', '-')} "
        f"| {sl.get('image_only', {}).get('f1_macro', '-')} "
        f"| {sl.get('both', {}).get('f1_macro', '-')} "
        f"| {rep.get('modality_robustness_gap', '-')} |"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--histories", nargs="+", required=True, help="JSON files of training history")
    ap.add_argument("--names", nargs="+", required=True, help="Labels matching --histories")
    ap.add_argument("--out", type=Path, default=Path("docs/baseline_report.md"))
    args = ap.parse_args()
    assert len(args.histories) == len(args.names)

    lines = [
        "# Baseline comparison",
        "",
        "| model | overall | text_only | image_only | both | robustness gap |",
        "|---|---|---|---|---|---|",
    ]
    for name, path in zip(args.names, args.histories):
        hist = json.loads(Path(path).read_text(encoding="utf-8"))
        best = max(hist, key=lambda r: r.get("overall", {}).get("f1_macro", -1))
        lines.append(_row(name, best))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] wrote {args.out}")


if __name__ == "__main__":
    main()
