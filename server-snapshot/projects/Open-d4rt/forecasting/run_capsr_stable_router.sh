#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/Open-d4rt
root_dir=/autodl-fs/data/d4rt_training/capsr_stable_router_20260828
mkdir -p "$root_dir"
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 \
  /root/miniconda3/envs/d4rt/bin/python forecasting/train_capsr_stable_router.py \
  --mode structured \
  --cache-root data/d4rt_forecasting_cache \
  --po-cache /autodl-fs/data/d4rt_datasets/d4rt_forecasting_cache_po50 \
  --po-split-manifest /autodl-fs/data/d4rt_datasets/worldtrack_pointodyssey50_v2/po50/split_manifest.json \
  --trajectory-checkpoint /autodl-fs/data/d4rt_training/pointodyssey50_uv_velocity_residual_60/best.pt \
  --backbone-checkpoint outputs/simvp_dynamic_ablation/no_geometry/best.pt \
  --init-checkpoint /autodl-fs/data/d4rt_training/capsr_v3_object_20260825/visibility000/best.pt \
  --freeze-legacy \
  --output-dir "$root_dir" \
  --epochs 10 --batch-size 4 --seed 42 --lr 3e-4 \
  --dynamic-weight 3.0 --visibility-weight 0.0 --temporal-weight 0.10 \
  --lpips-weight 0.03 --gradient-weight 0.05 \
  --static-identity-weight 0.10 --correction-weight 0.005 \
  --mask-budget-weight 0.002 --object-routing-weight 0.0 \
  --router-oracle-weight 0.01 --router-regret-weight 5.0 \
  --router-temporal-weight 0.03 --router-spatial-weight 0.02 \
  --router-prior-kl-weight 0.03 --router-region-kernel 5 \
  --router-benefit-margin 0.0005 \
  --router-structure-cost-weight 0.10 \
  --router-candidate-temporal-cost-weight 0.10 \
  --router-low-margin-temperature 0.001 \
  --motion-threshold 0.05 --num-motion-clusters 4 \
  --sigma 0.035 --hidden 48 --device cuda \
  >"$root_dir/run.log" 2>&1
date -Is >"$root_dir/COMPLETE"
