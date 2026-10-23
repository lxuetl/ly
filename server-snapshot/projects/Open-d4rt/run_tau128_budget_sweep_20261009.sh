#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/Open-d4rt
exec /root/miniconda3/envs/d4rt/bin/python -u train_tau128_scaled_cap.py \
  --d4rt-root /root/autodl-tmp/Open-d4rt \
  --cache-root data/d4rt_forecasting_cache \
  --po-cache /root/autodl-fs/d4rt_datasets/d4rt_forecasting_cache_po_full \
  --po-split-manifest /root/autodl-fs/d4rt_datasets/worldtrack_pointodyssey_full_v2/po50/split_manifest.json \
  --sample-manifest /root/autodl-fs/d4rt_training/tau_v2_direction_rank_pb002_20261009/split_paths.json \
  --backbone-checkpoint /root/autodl-fs/d4rt_training/tau128_full_fair_fast_20261009/best.pt \
  --trajectory-checkpoint /root/autodl-fs/d4rt_training/pointodyssey_full_uv_velocity_residual_60/best.pt \
  --output-dir /root/autodl-fs/d4rt_training/tau128_budget_sweep_matched_20261009 \
  --image-size 128 \
  --epochs 8 \
  --batch-size 2 \
  --lr 0.00005 \
  --flow-kernel 33 \
  --feature-cap 0.12 \
  --feature-budget 0.006 \
  --pixel-budget 0.002 \
  --budget-sweep 0.0006,0.0012,0.002 \
  --local-pixel-cap 0.08 \
  --pixel-budget-weight 1 \
  --direction-rank-weight 1 \
  --direction-rank-margin 0.000005 \
  --object-motion-threshold-px 4 \
  --object-mask-radius 6 \
  --split-seed 42 \
  --seed 42 \
  --num-workers 4
