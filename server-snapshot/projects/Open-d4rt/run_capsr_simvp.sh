#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/Open-d4rt

OUT=/autodl-fs/data/d4rt_training/capsr_simvp_full_20260824
mkdir -p "$OUT"

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 \
  /root/miniconda3/envs/d4rt/bin/python forecasting/train_capsr.py \
    --cache-root data/d4rt_forecasting_cache \
    --po-cache /autodl-fs/data/d4rt_datasets/d4rt_forecasting_cache_po50 \
    --po-split-manifest /autodl-fs/data/d4rt_datasets/worldtrack_pointodyssey50_v2/po50/split_manifest.json \
    --trajectory-checkpoint /autodl-fs/data/d4rt_training/pointodyssey50_uv_velocity_residual_60/best.pt \
    --backbone-checkpoint outputs/simvp_dynamic_ablation/no_geometry/best.pt \
    --output-dir "$OUT" \
    --epochs 20 --batch-size 4 \
    --dynamic-weight 3.0 --visibility-weight 0.15 \
    --temporal-weight 0.10 --lpips-weight 0.03 --gradient-weight 0.05 \
    --static-identity-weight 0.10 --correction-weight 0.005 \
    --sigma 0.035 --depth-temperature 1.0 --hidden 48 \
    --device cuda \
    >"$OUT/run.log" 2>&1
