#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/Open-d4rt
root_dir=/autodl-fs/data/d4rt_training/capsr_oracle_diagnosis_20260827
mkdir -p "$root_dir"
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 \
  /root/miniconda3/envs/d4rt/bin/python forecasting/evaluate_capsr_oracle_diagnosis.py \
  --cache-root data/d4rt_forecasting_cache \
  --po-cache /autodl-fs/data/d4rt_datasets/d4rt_forecasting_cache_po50 \
  --po-split-manifest /autodl-fs/data/d4rt_datasets/worldtrack_pointodyssey50_v2/po50/split_manifest.json \
  --trajectory-checkpoint /autodl-fs/data/d4rt_training/pointodyssey50_uv_velocity_residual_60/best.pt \
  --backbone-checkpoint outputs/simvp_dynamic_ablation/no_geometry/best.pt \
  --capsr-checkpoint /autodl-fs/data/d4rt_training/capsr_v3_object_20260825/visibility000/best.pt \
  --output-dir "$root_dir" --device cuda \
  >"$root_dir/run.log" 2>&1
date -Is >"$root_dir/COMPLETE"
