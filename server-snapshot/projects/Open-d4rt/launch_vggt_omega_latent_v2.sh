#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/Open-d4rt
PY_OMEGA=/root/autodl-tmp/vggt-omega/.venv/bin/python
PY_TRAIN=/root/miniconda3/envs/d4rt/bin/python
PO_CACHE=/root/autodl-fs/d4rt_datasets/d4rt_forecasting_cache_po_full
PO_SPLIT=/root/autodl-fs/d4rt_datasets/worldtrack_pointodyssey_full_v2/po50/split_manifest.json
OMEGA_WEIGHT=/root/autodl-fs/models/VGGT-Omega/vggt_omega_1b_512.pt
OMEGA_CACHE=/root/autodl-fs/d4rt_datasets/vggt_omega_latents_v2_20260930
BACKBONE=/root/autodl-fs/d4rt_training/four_backbone_fair64_bs2_20260927/tau/best.pt
OUTPUT=/root/autodl-fs/d4rt_training/vggt_omega_latent_v2_20260930

mkdir -p "$OMEGA_CACHE" "$OUTPUT"
cd "$ROOT"

"$PY_OMEGA" precompute_vggt_omega_latents_v2.py \
  --d4rt-root "$ROOT" \
  --cache-root data/d4rt_forecasting_cache \
  --po-cache "$PO_CACHE" \
  --po-split-manifest "$PO_SPLIT" \
  --checkpoint "$OMEGA_WEIGHT" \
  --output-root "$OMEGA_CACHE"

"$PY_TRAIN" train_vggt_omega_latent_v2.py \
  --d4rt-root "$ROOT" \
  --cache-root data/d4rt_forecasting_cache \
  --po-cache "$PO_CACHE" \
  --po-split-manifest "$PO_SPLIT" \
  --omega-cache "$OMEGA_CACHE" \
  --backbone-checkpoint "$BACKBONE" \
  --output-dir "$OUTPUT" \
  --image-size 64 \
  --epochs 12 \
  --batch-size 2 \
  --num-workers 2
