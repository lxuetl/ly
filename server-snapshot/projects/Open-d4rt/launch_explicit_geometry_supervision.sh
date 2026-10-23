#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0

ROOT=/root/autodl-tmp/Open-d4rt
PYTHON=/root/miniconda3/envs/d4rt/bin/python
DENSE=/autodl-fs/data/d4rt_datasets/vggt_dense_static_20260924
OUTPUT=/autodl-fs/data/d4rt_training/explicit_vggt_static_d4rt_dynamic_20260924
COMMON=(
  --d4rt-root "$ROOT"
  --cache-root data/d4rt_forecasting_cache
  --po-cache /autodl-fs/data/d4rt_datasets/d4rt_forecasting_cache_po_full
  --po-split-manifest /autodl-fs/data/d4rt_datasets/worldtrack_pointodyssey_full_v2/po50/split_manifest.json
  --legacy-subsets adt_mini,ds_mini,po_mini,pstudio_mini
  --max-train 216 --max-val 80 --max-test 80
)

cd "$ROOT"
mkdir -p "$DENSE" "$OUTPUT"
if [[ ! -f "$DENSE/COMPLETE" ]]; then
  "$PYTHON" precompute_vggt_dense_static.py "${COMMON[@]}" \
    --checkpoint /autodl-fs/data/models/VGGT-Omega/vggt_omega_1b_512.pt \
    --output-root "$DENSE" \
    --resolution 512 --output-size 64
fi

"$PYTHON" train_explicit_geometry_supervision.py "${COMMON[@]}" \
  --dense-geometry-cache "$DENSE" \
  --backbone-checkpoint /autodl-fs/data/d4rt_training/simvp_fair_po_full_20260922/best.pt \
  --output-dir "$OUTPUT" \
  --epochs 16 --batch-size 2 \
  --static-weight 0.10 --d4rt-weight 0.25 \
  --lpips-weight 0.03 --temporal-weight 0.10
