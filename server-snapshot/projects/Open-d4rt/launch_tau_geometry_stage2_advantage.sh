#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/Open-d4rt
PYTHON=/root/miniconda3/envs/d4rt/bin/python
OUTPUT=/autodl-fs/data/d4rt_training/tau_geometry_stage2_advantage_full_20260930

mkdir -p "$OUTPUT"
cd "$ROOT"
exec "$PYTHON" train_tau_geometry_stage2_advantage.py \
  --d4rt-root "$ROOT" \
  --cache-root data/d4rt_forecasting_cache \
  --po-cache /autodl-fs/data/d4rt_datasets/d4rt_forecasting_cache_po_full \
  --po-split-manifest /autodl-fs/data/d4rt_datasets/worldtrack_pointodyssey_full_v2/po50/split_manifest.json \
  --backbone-checkpoint /autodl-fs/data/d4rt_training/four_backbone_fair64_bs2_20260927/tau/best.pt \
  --trajectory-checkpoint /autodl-fs/data/d4rt_training/pointodyssey_full_uv_velocity_residual_60/best.pt \
  --control-checkpoint /autodl-fs/data/d4rt_training/tau_lastblock_geometry_v3_full_20260929/control_best.pt \
  --output-dir "$OUTPUT" \
  --image-size 64 \
  --epochs 12 \
  --batch-size 2 \
  --adapter-lr 3e-4 \
  --static-weight 0.50 \
  --advantage-weight 0.50 \
  --wrong-rank-weight 0.50 \
  --wrong-reject-weight 0.10 \
  --advantage-margin 0.001 \
  --wrong-margin 0.001 \
  --selection-regression-weight 2.0 \
  --seed 42 \
  --num-workers 4
