#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/Open-d4rt
PYTHON=/root/miniconda3/envs/d4rt/bin/python
OMEGA_CHECKPOINT=/root/autodl-fs/models/VGGT-Omega/vggt_omega_1b_512.pt
SPATIAL_CACHE=/root/autodl-fs/d4rt_datasets/vggt_omega_spatial_v4_20261002
OUTPUT=/root/autodl-fs/d4rt_training/vggt_omega_spatial_v4_20261002
PO_CACHE=/root/autodl-fs/d4rt_datasets/d4rt_forecasting_cache_po_full
PO_SPLIT=/root/autodl-fs/d4rt_datasets/worldtrack_pointodyssey_full_v2/po50/split_manifest.json
TAU_CHECKPOINT=/root/autodl-fs/d4rt_training/four_backbone_fair64_bs2_20260927/tau/best.pt

mkdir -p "$SPATIAL_CACHE" "$OUTPUT"
cd "$ROOT"

if [[ ! -f "$SPATIAL_CACHE/COMPLETE" ]]; then
  "$PYTHON" precompute_vggt_omega_spatial_v4.py \
    --d4rt-root "$ROOT" \
    --cache-root data/d4rt_forecasting_cache \
    --po-cache "$PO_CACHE" \
    --po-split-manifest "$PO_SPLIT" \
    --checkpoint "$OMEGA_CHECKPOINT" \
    --output-root "$SPATIAL_CACHE" \
    --resolution 512 \
    --output-size 64 \
    --patch-grid 8
fi

"$PYTHON" train_vggt_omega_spatial_v4.py \
  --d4rt-root "$ROOT" \
  --cache-root data/d4rt_forecasting_cache \
  --po-cache "$PO_CACHE" \
  --po-split-manifest "$PO_SPLIT" \
  --omega-cache "$SPATIAL_CACHE" \
  --backbone-checkpoint "$TAU_CHECKPOINT" \
  --output-dir "$OUTPUT" \
  --alignment-epochs 4 \
  --epochs 12 \
  --batch-size 2 \
  --semantic-dim 192 \
  --num-workers 2
