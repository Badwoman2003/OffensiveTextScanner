# Training Pipeline

OTS uses a three-stage training recipe. Every stage emits a checkpoint that the next stage can
consume so you can start or resume from any point.

## Stage-A — Chinese Continued Pretraining (CPT)

Script: `ots_core/training/stage_a_cpt.py`

- Starts from `dandelin/vilt-b32-mlm`.
- Tokenizer swapped to `bert-base-chinese`; word-embedding table warm-started by copying the
  English mean row into every new Chinese row (see `warm_start_embeddings`).
- Objectives: **MLM** (15 % masking) + **ITM** (50 % in-batch swap) + **WPA** (IPOT over last
  hidden states, `ots_core/training/wpa.py`).

Recommended corpora:

- Wukong-1G (`https://wukong-dataset.github.io/`)
- Laion-400M Chinese slice
- CC3M machine-translated to zh

Convert a source into a parquet with columns `image_path, text` and run:

```bash
python -m ots_core.training.stage_a_cpt --pairs data/processed/cc3m_zh.parquet \
                                          --out checkpoints/vilt-zh-cpt --epochs 3 --fp16
```

## Stage-B — Supervised Fine-Tuning

Script: `ots_core/training/stage_b_finetune.py`

- Mixed dataset: `coldataset_*` (text-only) ∪ `hateful_memes_*` + `chinese_memes_*` + synthetic
  `text_render_*` (both).
- Sampler: `StratifiedModalitySampler` (effective-number weights over
  `source × modality × label`).
- Loss: Focal (γ=2) by default; CE selectable.
- Reports per-slice F1 and `modality_robustness_gap` each epoch.

```bash
python -m ots_core.training.stage_b_finetune \
    --train data/processed/train_mixed.parquet \
    --val   data/processed/val_mixed.parquet   \
    --init  checkpoints/vilt-zh-cpt/final \
    --out   checkpoints/vilt-offense-stageB --epochs 4 --fp16
```

## Stage-C — Modality Imbalance Specialist

Script: `ots_core/training/stage_c_imbalance.py`

- Attaches auxiliary `ModalityHead`s and runs every batch through three views of the backbone
  (text-only, image-only, multimodal).
- Optimises `GradientBlendedLoss`; after each epoch the weights are recomputed via
  `solve_weights(text_stats, image_stats, multi_stats)` (see `gradient_blending.py`).

```bash
python -m ots_core.training.stage_c_imbalance \
    --train data/processed/train_mixed.parquet \
    --val   data/processed/val_mixed.parquet   \
    --init  checkpoints/vilt-offense-stageB/best \
    --out   checkpoints/vilt-offense-final --epochs 3
```

## Evaluation & baselining

```bash
python scripts/build_baseline_report.py \
    --histories checkpoints/legacy-bert/history.json \
                checkpoints/vilt-offense-stageB/history.json \
                checkpoints/vilt-offense-final/history.json \
    --names "Legacy BERT (text)" "ViLT Stage-B" "ViLT Stage-C (GB)" \
    --out   docs/baseline_report.md
```

The legacy BERT+PaddleOCR pipeline lives in `legacy/OffensiveTextScanner` so you can rerun it
for an apples-to-oranges text-only comparison if needed.
