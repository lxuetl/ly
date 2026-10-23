#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0

ROOT=/root/autodl-tmp/Open-d4rt
PYTHON=/root/miniconda3/envs/d4rt/bin/python
DENSE=/autodl-fs/data/d4rt_datasets/vggt_dense_static_20260924
OUTPUT=/autodl-fs/data/d4rt_training/masked_dynamic_supervision_20260925

test -f "$DENSE/COMPLETE"
mkdir -p "$OUTPUT"
cd "$ROOT"

"$PYTHON" train_explicit_geometry_supervision.py \
  --d4rt-root "$ROOT" \
  --cache-root data/d4rt_forecasting_cache \
  --po-cache /autodl-fs/data/d4rt_datasets/d4rt_forecasting_cache_po_full \
  --po-split-manifest /autodl-fs/data/d4rt_datasets/worldtrack_pointodyssey_full_v2/po50/split_manifest.json \
  --legacy-subsets adt_mini,ds_mini,po_mini,pstudio_mini \
  --dense-geometry-cache "$DENSE" \
  --backbone-checkpoint /autodl-fs/data/d4rt_training/simvp_fair_po_full_20260922/best.pt \
  --output-dir "$OUTPUT" \
  --epochs 16 --batch-size 2 \
  --static-objective protect --include-d4rt-only \
  --static-weight 0.10 --d4rt-weight 0.25 \
  --lpips-weight 0.03 --temporal-weight 0.10 \
  --max-train 216 --max-val 80 --max-test 80
