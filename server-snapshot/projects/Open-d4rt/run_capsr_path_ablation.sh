#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/Open-d4rt

OUT=/autodl-fs/data/d4rt_training/capsr_path_ablation_full_20260825
mkdir -p "$OUT"

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /root/miniconda3/envs/d4rt/bin/python forecasting/evaluate_capsr_paths.py \
    --cache-root data/d4rt_forecasting_cache \
    --po-cache /autodl-fs/data/d4rt_datasets/d4rt_forecasting_cache_po50 \
    --po-split-manifest /autodl-fs/data/d4rt_datasets/worldtrack_pointodyssey50_v2/po50/split_manifest.json \
    --trajectory-checkpoint /autodl-fs/data/d4rt_training/pointodyssey50_uv_velocity_residual_60/best.pt \
    --backbone-checkpoint outputs/simvp_dynamic_ablation/no_geometry/best.pt \
    --capsr-checkpoint /autodl-fs/data/d4rt_training/capsr_simvp_full_20260824/best.pt \
    --output-dir "$OUT" \
    --device cuda \
    >"$OUT/run.log" 2>&1
