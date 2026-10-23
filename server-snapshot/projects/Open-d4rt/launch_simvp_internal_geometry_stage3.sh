#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/Open-d4rt}"
PYTHON="${PYTHON:-/root/miniconda3/envs/d4rt/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-/autodl-fs/data/d4rt_training/simvp_internal_geometry_stage3_20260925}"

cd "$ROOT"
mkdir -p "$OUTPUT_DIR"
export OMP_NUM_THREADS=4

exec "$PYTHON" train_simvp_internal_geometry_stage3.py \
  --d4rt-root "$ROOT" \
  --cache-root data/d4rt_forecasting_cache \
  --po-cache /autodl-fs/data/d4rt_datasets/d4rt_forecasting_cache_po_full \
  --po-split-manifest /autodl-fs/data/d4rt_datasets/worldtrack_pointodyssey_full_v2/po50/split_manifest.json \
  --sample-manifest /autodl-fs/data/d4rt_datasets/vggt_dense_static_20260924/manifest.json \
  --backbone-checkpoint /autodl-fs/data/d4rt_training/simvp_fair_po_full_20260922/best.pt \
  --trajectory-checkpoint /autodl-fs/data/d4rt_training/pointodyssey_full_uv_velocity_residual_60/best.pt \
  --v1-checkpoint /autodl-fs/data/d4rt_training/simvp_feature_warp_round1_20260925/correct_best.pt \
  --output-dir "$OUTPUT_DIR" \
  --epochs "${EPOCHS:-8}" \
  --batch-size "${BATCH_SIZE:-2}" \
  --max-train "${MAX_TRAIN:-216}" \
  --max-val "${MAX_VAL:-80}" \
  --max-test "${MAX_TEST:-80}"
