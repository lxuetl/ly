#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/Open-d4rt
exec /root/miniconda3/envs/d4rt/bin/python -u train_tau128_local_residual_diffusion.py \
  --d4rt-root /root/autodl-tmp/Open-d4rt \
  --cache-root data/d4rt_forecasting_cache \
  --po-cache /root/autodl-fs/d4rt_datasets/d4rt_forecasting_cache_po_full \
  --po-split-manifest /root/autodl-fs/d4rt_datasets/worldtrack_pointodyssey_full_v2/po50/split_manifest.json \
  --sample-manifest /root/autodl-fs/d4rt_training/tau_v2_direction_rank_pb002_20261009/split_paths.json \
  --backbone-checkpoint /root/autodl-fs/d4rt_training/tau128_full_fair_fast_20261009/best.pt \
  --trajectory-checkpoint /root/autodl-fs/d4rt_training/pointodyssey_full_uv_velocity_residual_60/best.pt \
  --geometry-checkpoint /root/autodl-fs/d4rt_training/tau128_budget_sweep_matched_20261009/geometry_b2000_best.pt \
  --output-dir /root/autodl-fs/d4rt_training/tau128_local_residual_diffusion_20261010 \
  --epochs 6 \
  --batch-size 2 \
  --num-workers 4 \
  --seed 42 \
  "$@"
