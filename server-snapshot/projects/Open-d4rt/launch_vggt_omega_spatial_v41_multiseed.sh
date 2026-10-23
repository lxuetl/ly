#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

ROOT=/root/autodl-tmp/Open-d4rt
PYTHON=/root/miniconda3/envs/d4rt/bin/python
SPATIAL_CACHE=/root/autodl-fs/d4rt_datasets/vggt_omega_spatial_v4_20261002
OUTPUT_ROOT=/root/autodl-fs/d4rt_training/vggt_omega_spatial_v41_20261004
PO_CACHE=/root/autodl-fs/d4rt_datasets/d4rt_forecasting_cache_po_full
PO_SPLIT=/root/autodl-fs/d4rt_datasets/worldtrack_pointodyssey_full_v2/po50/split_manifest.json
TAU_CHECKPOINT=/root/autodl-fs/d4rt_training/four_backbone_fair64_bs2_20260927/tau/best.pt

if [[ ! -f "$SPATIAL_CACHE/COMPLETE" ]]; then
  echo "spatial cache is incomplete: $SPATIAL_CACHE" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"
cd "$ROOT"

for SEED in 42 123 3407; do
  OUTPUT="$OUTPUT_ROOT/seed_$SEED"
  mkdir -p "$OUTPUT"
  if [[ -f "$OUTPUT/COMPLETE" ]]; then
    echo "seed $SEED already complete"
    continue
  fi
  "$PYTHON" train_vggt_omega_spatial_v41.py \
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
    --min-match-gap 0.03 \
    --seed "$SEED" \
    --num-workers 2
done

"$PYTHON" aggregate_vggt_omega_spatial_v41.py \
  --root "$OUTPUT_ROOT" \
  --seeds 42,123,3407
