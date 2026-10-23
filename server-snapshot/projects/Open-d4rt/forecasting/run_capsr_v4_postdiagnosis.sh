#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/Open-d4rt
TRAIN=/autodl-fs/data/d4rt_training/capsr_v4_singleconf_adaptive_20260825
OUT=/autodl-fs/data/d4rt_training/capsr_v4_path_diagnosis_20260825
mkdir -p "$OUT"

while [[ ! -f "$TRAIN/COMPLETE" ]]; do
  sleep 30
done

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 \
  /root/miniconda3/envs/d4rt/bin/python forecasting/evaluate_capsr_v3_diagnostics.py \
    --cache-root data/d4rt_forecasting_cache \
    --po-cache /autodl-fs/data/d4rt_datasets/d4rt_forecasting_cache_po50 \
    --po-split-manifest /autodl-fs/data/d4rt_datasets/worldtrack_pointodyssey50_v2/po50/split_manifest.json \
    --trajectory-checkpoint /autodl-fs/data/d4rt_training/pointodyssey50_uv_velocity_residual_60/best.pt \
    --backbone-checkpoint outputs/simvp_dynamic_ablation/no_geometry/best.pt \
    --correction-checkpoint "$TRAIN/best.pt" \
    --output "$OUT/summary.json" --device cuda \
    >"$OUT/run.log" 2>&1

date -Is >"$OUT/COMPLETE"
