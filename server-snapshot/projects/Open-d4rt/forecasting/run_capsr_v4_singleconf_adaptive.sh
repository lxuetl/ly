#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/Open-d4rt
OUT=/autodl-fs/data/d4rt_training/capsr_v4_singleconf_adaptive_20260825
mkdir -p "$OUT"

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 \
  /root/miniconda3/envs/d4rt/bin/python forecasting/train_capsr_structured.py \
    --mode structured \
    --cache-root data/d4rt_forecasting_cache \
    --po-cache /autodl-fs/data/d4rt_datasets/d4rt_forecasting_cache_po50 \
    --po-split-manifest /autodl-fs/data/d4rt_datasets/worldtrack_pointodyssey50_v2/po50/split_manifest.json \
    --trajectory-checkpoint /autodl-fs/data/d4rt_training/pointodyssey50_uv_velocity_residual_60/best.pt \
    --backbone-checkpoint outputs/simvp_dynamic_ablation/no_geometry/best.pt \
    --output-dir "$OUT" \
    --epochs 20 --batch-size 4 --seed 42 \
    --dynamic-weight 3.0 --visibility-weight 0 \
    --temporal-weight 0.10 --lpips-weight 0.03 --gradient-weight 0.05 \
    --static-identity-weight 0.10 --correction-weight 0.005 \
    --mask-budget-weight 0.005 --object-routing-weight 0.05 \
    --motion-threshold 0.05 --num-motion-clusters 4 \
    --sigma 0.035 --hidden 48 --device cuda \
    >"$OUT/run.log" 2>&1

date -Is >"$OUT/COMPLETE"
