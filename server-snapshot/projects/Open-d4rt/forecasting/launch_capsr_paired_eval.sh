#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/Open-d4rt
OUT=/autodl-fs/data/d4rt_training/capsr_paired_eval_20260828
TRAIN=/autodl-fs/data/d4rt_training
PY=/root/miniconda3/envs/d4rt/bin/python
mkdir -p "$OUT"

COMMON=(
  --cache-root data/d4rt_forecasting_cache
  --po-cache /autodl-fs/data/d4rt_datasets/d4rt_forecasting_cache_po50
  --po-split-manifest /autodl-fs/data/d4rt_datasets/worldtrack_pointodyssey50_v2/po50/split_manifest.json
  --trajectory-checkpoint "$TRAIN/pointodyssey50_uv_velocity_residual_60/best.pt"
  --backbone-checkpoint outputs/simvp_dynamic_ablation/no_geometry/best.pt
  --motion-threshold 0.05 --seed 42 --device cuda
)

CUDA_VISIBLE_DEVICES=0 "$PY" forecasting/evaluate_capsr_per_sequence.py \
  --variant v5 --checkpoint "$TRAIN/capsr_calibrated_v5_20260827/best.pt" \
  --output "$OUT/v5.json" "${COMMON[@]}" >"$OUT/v5.log" 2>&1 & p1=$!
CUDA_VISIBLE_DEVICES=0 "$PY" forecasting/evaluate_capsr_per_sequence.py \
  --variant router_v1 --checkpoint "$TRAIN/capsr_multipath_router_20260827/best.pt" \
  --output "$OUT/router_v1.json" "${COMMON[@]}" >"$OUT/router_v1.log" 2>&1 & p2=$!
CUDA_VISIBLE_DEVICES=0 "$PY" forecasting/evaluate_capsr_per_sequence.py \
  --variant stable_v2 --checkpoint "$TRAIN/capsr_stable_router_20260828/best.pt" \
  --output "$OUT/stable_v2.json" "${COMMON[@]}" >"$OUT/stable_v2.log" 2>&1 & p3=$!

wait "$p1" "$p2" "$p3"
"$PY" forecasting/aggregate_capsr_paired.py \
  --inputs "$OUT/v5.json" "$OUT/router_v1.json" "$OUT/stable_v2.json" \
  --output-dir "$OUT" --bootstrap 10000 --seed 42 >"$OUT/aggregate.log" 2>&1
date -Is >"$OUT/COMPLETE"
