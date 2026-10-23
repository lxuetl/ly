#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/Open-d4rt

ROOT=/autodl-fs/data/d4rt_training/stage3_temporal3d_risk_gate_20260823
mkdir -p "$ROOT"

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0 \
  /root/miniconda3/envs/d4rt/bin/python forecasting/train_pointodyssey_dmvfn_injection.py \
  --cache-root data/d4rt_forecasting_cache \
  --po-cache /autodl-fs/data/d4rt_datasets/d4rt_forecasting_cache_po50 \
  --po-split-manifest /autodl-fs/data/d4rt_datasets/worldtrack_pointodyssey50_v2/po50/split_manifest.json \
  --trajectory-checkpoint /autodl-fs/data/d4rt_training/pointodyssey50_uv_velocity_residual_60/best.pt \
  --backbone dmvfn --backbone-checkpoint outputs/dmvfn_motionloss_60/no_geometry/best.pt \
  --fusion-type feature_proposal --feature-channels 16 --proposal-residual-scale 0.08 \
  --init-corrector /autodl-fs/data/d4rt_training/stage2_dual_benefit_gate_20260823/reliable_geometry_motion_comp/best.pt \
  --benefit-gate-bias 0.1 --benefit-gate-warmup-epochs 2 \
  --dual-benefit-head --temporal-risk-margin 0.0005 \
  --temporal-risk-patch-kernel 3 --temporal-risk-positive-weight 1.25 \
  --temporal-risk-prob-floor 0.55 --temporal-risk-recall-loss-weight 0.50 \
  --temporal-risk-loss-weight 0.60 --joint-benefit-loss-weight 0.10 \
  --temporal-risk-recall-floor 0.35 --temporal-risk-recall-penalty 0.003 \
  --benefit-loss-weight 0.50 --benefit-margin 0.002 \
  --benefit-negative-weight 1.0 --benefit-temperature 0.01 \
  --benefit-positive-prob-floor 0.52 --benefit-recall-loss-weight 0.5 \
  --benefit-patch-kernel 7 --benefit-budget-weight 1.0 \
  --benefit-topk-fraction 0.30 --benefit-min-joint-probability 0.20 \
  --temporal-risk-max-probability 0.55 --benefit-temporal-align-weight 1.0 \
  --benefit-recall-floor 0.20 --benefit-recall-penalty 0.01 \
  --epochs 12 --batch-size 4 \
  --min-motion 0.025 --min-reliability 0.30 --mask-power 1.5 --dilation-kernel 1 \
  --base-motion-threshold 0.005 --base-motion-ramp 0.04 \
  --first-step-scale 0.5 --horizon-ramp-steps 4 \
  --geometry-source predicted --geometry-features uv_only \
  --lpips-loss-weight 0.03 --temporal-loss-weight 0.15 --gradient-loss-weight 0.05 \
  --static-identity-weight 0.10 \
  --temporal-select-weight 0.03 --static-select-weight 0.02 --mae-select-weight 0.005 \
  --device cuda --output-dir "$ROOT"
