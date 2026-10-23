#!/usr/bin/env bash
set -euo pipefail

PY=/root/miniconda3/envs/d4rt/bin/python
PO_CACHE=/autodl-fs/data/d4rt_datasets/d4rt_forecasting_cache_po50
PO_SPLIT=/autodl-fs/data/d4rt_datasets/worldtrack_pointodyssey50_v2/po50/split_manifest.json
OLD_TRAJ=/autodl-fs/data/d4rt_training/pointodyssey50_curriculum_80/best.pt
UV_OUT=/autodl-fs/data/d4rt_training/pointodyssey50_uv_velocity_residual_60
VIDEO_OUT=/autodl-fs/data/d4rt_training/po50_dmvfn_uv_perceptual_tight_30

mkdir -p "$UV_OUT" "$VIDEO_OUT"

"$PY" forecasting/train_pointodyssey_joint.py \
  --po-cache "$PO_CACHE" \
  --po-split-manifest "$PO_SPLIT" \
  --output-dir "$UV_OUT" \
  --epochs 60 --batch-size 2 --device cuda \
  --model-variant velocity_residual \
  --init-checkpoint "$OLD_TRAJ"

"$PY" forecasting/train_pointodyssey_dmvfn_injection.py \
  --po-cache "$PO_CACHE" \
  --po-split-manifest "$PO_SPLIT" \
  --trajectory-checkpoint "$UV_OUT/best.pt" \
  --backbone-checkpoint outputs/dmvfn_motionloss_60/no_geometry/best.pt \
  --init-corrector /autodl-fs/data/d4rt_training/po50_dmvfn_ablation_corrected/xyz_uv/reliable_geometry_motion_comp/best.pt \
  --output-dir "$VIDEO_OUT" \
  --epochs 30 --batch-size 2 --device cuda \
  --geometry-source predicted --geometry-features uv_only \
  --min-motion 0.025 --min-reliability 0.30 \
  --mask-power 1.5 --dilation-kernel 1 \
  --lpips-loss-weight 0.03 \
  --temporal-loss-weight 0.10 \
  --gradient-loss-weight 0.05 \
  --temporal-select-weight 0.02
