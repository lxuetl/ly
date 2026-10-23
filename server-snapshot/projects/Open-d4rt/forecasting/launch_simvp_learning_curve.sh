#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/Open-d4rt
ROOT=/autodl-fs/data/d4rt_training/simvp_fair_learning_curve_20260828
PY=/root/miniconda3/envs/d4rt/bin/python
mkdir -p "$ROOT"

run_fraction() {
  local tag=$1
  local fraction=$2
  mkdir -p "$ROOT/$tag"
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$PY" \
    forecasting/train_simvp_fair_combined.py \
    --cache-root data/d4rt_forecasting_cache \
    --po-cache /autodl-fs/data/d4rt_datasets/d4rt_forecasting_cache_po50 \
    --po-split-manifest /autodl-fs/data/d4rt_datasets/worldtrack_pointodyssey50_v2/po50/split_manifest.json \
    --output-dir "$ROOT/$tag" --train-fraction "$fraction" \
    --epochs 30 --patience 8 --batch-size 4 --lr 3e-4 --seed 42 --device cuda \
    >"$ROOT/$tag/run.log" 2>&1
}

run_fraction fraction025 0.25 & p1=$!
run_fraction fraction050 0.50 & p2=$!
run_fraction fraction100 1.00 & p3=$!
wait "$p1" "$p2" "$p3"
date -Is >"$ROOT/COMPLETE"
