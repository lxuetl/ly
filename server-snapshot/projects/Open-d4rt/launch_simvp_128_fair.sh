#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/Open-d4rt}"
PYTHON="${PYTHON:-/root/miniconda3/envs/d4rt/bin/python}"
BASE_DIR="${BASE_DIR:-/autodl-fs/data/d4rt_training/simvp_128_fraction025_20260926}"
V1_DIR="${V1_DIR:-/autodl-fs/data/d4rt_training/simvp_v1_128_fraction025_20260926}"

cd "$ROOT"
mkdir -p "$BASE_DIR" "$V1_DIR"
export OMP_NUM_THREADS=4

if [[ ! -f "$BASE_DIR/COMPLETE" ]]; then
  "$PYTHON" train_simvp_fair_combined.py \
    --cache-root data/d4rt_forecasting_cache \
    --po-cache /autodl-fs/data/d4rt_datasets/d4rt_forecasting_cache_po_full \
    --po-split-manifest /autodl-fs/data/d4rt_datasets/worldtrack_pointodyssey_full_v2/po50/split_manifest.json \
    --output-dir "$BASE_DIR" \
    --train-fraction 0.25 \
    --image-size 128 \
    --max-val 80 \
    --max-test 80 \
    --epochs "${BASE_EPOCHS:-15}" \
    --patience 5 \
    --batch-size "${BASE_BATCH_SIZE:-4}"
fi

"$PYTHON" train_simvp_feature_warp_128.py \
  --d4rt-root "$ROOT" \
  --cache-root data/d4rt_forecasting_cache \
  --po-cache /autodl-fs/data/d4rt_datasets/d4rt_forecasting_cache_po_full \
  --po-split-manifest /autodl-fs/data/d4rt_datasets/worldtrack_pointodyssey_full_v2/po50/split_manifest.json \
  --sample-manifest "$BASE_DIR/split_paths.json" \
  --backbone-checkpoint "$BASE_DIR/best.pt" \
  --trajectory-checkpoint /autodl-fs/data/d4rt_training/pointodyssey_full_uv_velocity_residual_60/best.pt \
  --output-dir "$V1_DIR" \
  --image-size 128 \
  --epochs "${V1_EPOCHS:-8}" \
  --batch-size "${V1_BATCH_SIZE:-2}" \
  --flow-kernel 17
