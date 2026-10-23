#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/Open-d4rt

if ! /root/miniconda3/envs/d4rt/bin/python -c 'import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)'; then
  echo "CUDA GPU is required for stage-3 training; no GPU is mounted." >&2
  exit 2
fi

OUT=/autodl-fs/data/d4rt_training/stage3_feature_proposal_benefit_b4e20
mkdir -p "$OUT"

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
  /root/miniconda3/envs/d4rt/bin/python forecasting/train_pointodyssey_dmvfn_injection.py \
  --cache-root data/d4rt_forecasting_cache \
  --po-cache /autodl-fs/data/d4rt_datasets/d4rt_forecasting_cache_po50 \
  --po-split-manifest /autodl-fs/data/d4rt_datasets/worldtrack_pointodyssey50_v2/po50/split_manifest.json \
  --trajectory-checkpoint /autodl-fs/data/d4rt_training/pointodyssey50_uv_velocity_residual_60/best.pt \
  --backbone-checkpoint outputs/dmvfn_motionloss_60/no_geometry/best.pt \
  --output-dir "$OUT" \
  --fusion-type feature_proposal \
  --feature-channels 16 \
  --proposal-residual-scale 0.08 \
  --epochs 20 --batch-size 4 \
  --min-motion 0.025 --min-reliability 0.30 --mask-power 1.5 --dilation-kernel 1 \
  --base-motion-threshold 0.005 --base-motion-ramp 0.04 \
  --first-step-scale 0.5 --horizon-ramp-steps 4 \
  --geometry-source predicted --geometry-features uv_only \
  --benefit-loss-weight 0.10 --benefit-margin 0.002 --benefit-negative-weight 1.5 \
  --lpips-loss-weight 0.03 --temporal-loss-weight 0.15 --gradient-loss-weight 0.05 \
  --static-identity-weight 0.10 \
  --temporal-select-weight 0.03 --static-select-weight 0.02 --mae-select-weight 0.005 \
  --device cuda
