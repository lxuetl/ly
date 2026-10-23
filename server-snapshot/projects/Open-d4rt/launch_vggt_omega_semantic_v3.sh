#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/Open-d4rt
PYTHON=/root/miniconda3/envs/d4rt/bin/python
OUTPUT=/root/autodl-fs/d4rt_training/vggt_omega_semantic_v3_20261001

mkdir -p "$OUTPUT"
cd "$ROOT"

"$PYTHON" train_vggt_omega_semantic_v3.py \
  --d4rt-root "$ROOT" \
  --cache-root data/d4rt_forecasting_cache \
  --po-cache /root/autodl-fs/d4rt_datasets/d4rt_forecasting_cache_po_full \
  --po-split-manifest /root/autodl-fs/d4rt_datasets/worldtrack_pointodyssey_full_v2/po50/split_manifest.json \
  --omega-cache /root/autodl-fs/d4rt_datasets/vggt_omega_latents_v2_20260930 \
  --backbone-checkpoint /root/autodl-fs/d4rt_training/four_backbone_fair64_bs2_20260927/tau/best.pt \
  --output-dir "$OUTPUT" \
  --alignment-epochs 4 \
  --epochs 12 \
  --batch-size 2 \
  --num-workers 2
