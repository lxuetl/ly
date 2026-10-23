#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/Open-d4rt

ROOT=/autodl-fs/data/d4rt_training/capsr_v3_object_20260825
COMMON=(
  --mode structured
  --cache-root data/d4rt_forecasting_cache
  --po-cache /autodl-fs/data/d4rt_datasets/d4rt_forecasting_cache_po50
  --po-split-manifest /autodl-fs/data/d4rt_datasets/worldtrack_pointodyssey50_v2/po50/split_manifest.json
  --trajectory-checkpoint /autodl-fs/data/d4rt_training/pointodyssey50_uv_velocity_residual_60/best.pt
  --backbone-checkpoint outputs/simvp_dynamic_ablation/no_geometry/best.pt
  --epochs 20 --batch-size 4 --seed 42
  --dynamic-weight 3.0 --temporal-weight 0.10
  --lpips-weight 0.03 --gradient-weight 0.05
  --static-identity-weight 0.10 --correction-weight 0.005
  --mask-budget-weight 0.005 --object-routing-weight 0.05
  --motion-threshold 0.05 --num-motion-clusters 4
  --sigma 0.035 --hidden 48 --device cuda
)

mkdir -p "$ROOT/visibility015" "$ROOT/visibility000"

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 \
  /root/miniconda3/envs/d4rt/bin/python forecasting/train_capsr_structured.py \
  --visibility-weight 0.15 --output-dir "$ROOT/visibility015" "${COMMON[@]}" \
  >"$ROOT/visibility015/run.log" 2>&1 &
PID_VIS=$!

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 \
  /root/miniconda3/envs/d4rt/bin/python forecasting/train_capsr_structured.py \
  --visibility-weight 0.0 --output-dir "$ROOT/visibility000" "${COMMON[@]}" \
  >"$ROOT/visibility000/run.log" 2>&1 &
PID_NOVIS=$!

printf '%s\n' "$PID_VIS" >"$ROOT/visibility015/pid"
printf '%s\n' "$PID_NOVIS" >"$ROOT/visibility000/pid"

wait "$PID_VIS"
wait "$PID_NOVIS"
date -Is >"$ROOT/COMPLETE"
