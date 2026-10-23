#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

ROOT=/root/autodl-tmp/Open-d4rt
PYTHON=/root/miniconda3/envs/d4rt/bin/python
OUTPUT=/root/autodl-fs/d4rt_training/tau_uv_warp_causal_v1_20261006

mkdir -p "$OUTPUT"
cd "$ROOT"

"$PYTHON" train_tau_uv_warp_causal_v1.py \
  --d4rt-root "$ROOT" \
  --cache-root data/d4rt_forecasting_cache \
  --po-cache /root/autodl-fs/d4rt_datasets/d4rt_forecasting_cache_po_full \
  --po-split-manifest /root/autodl-fs/d4rt_datasets/worldtrack_pointodyssey_full_v2/po50/split_manifest.json \
  --backbone-checkpoint /root/autodl-fs/d4rt_training/four_backbone_fair64_bs2_20260927/tau/best.pt \
  --trajectory-checkpoint /root/autodl-fs/d4rt_training/pointodyssey_full_uv_velocity_residual_60/best.pt \
  --output-dir "$OUTPUT" \
  --epochs 16 \
  --batch-size 2 \
  --split-seed 42 \
  --seed 42 \
  --num-workers 2
