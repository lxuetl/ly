#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/Open-d4rt
PYTHON=/root/miniconda3/envs/d4rt/bin/python
DATA=/root/autodl-fs/d4rt_datasets
RUNS=/root/autodl-fs/d4rt_training
SPLIT="$RUNS/tau_v2_direction_rank_pb002_20261009/split_paths.json"
BASE="$RUNS/tau128_full_fair_fast_20261009"
ADAPTER="$RUNS/tau128_v2_direction_rank_fast_20261009"
mkdir -p "$BASE" "$ADAPTER"

echo "$(date -Is) stage=backbone_start"
"$PYTHON" train_four_backbone_fair_128fast.py \
  --backbone tau \
  --cache-root data/d4rt_forecasting_cache \
  --po-cache "$DATA/d4rt_forecasting_cache_po_full" \
  --po-split-manifest "$DATA/worldtrack_pointodyssey_full_v2/po50/split_manifest.json" \
  --sample-manifest "$SPLIT" \
  --output-dir "$BASE" \
  --image-size 128 --epochs 15 --patience 5 --batch-size 2 --num-workers 4 --lr 0.0003 \
  > "$BASE/train.log" 2>&1
test -s "$BASE/COMPLETE"
test -s "$BASE/best.pt"
echo "$(date -Is) stage=backbone_complete"

echo "$(date -Is) stage=adapter_start"
"$PYTHON" train_tau_v2_direction_rank.py \
  --d4rt-root /root/autodl-tmp/Open-d4rt \
  --cache-root data/d4rt_forecasting_cache \
  --po-cache "$DATA/d4rt_forecasting_cache_po_full" \
  --po-split-manifest "$DATA/worldtrack_pointodyssey_full_v2/po50/split_manifest.json" \
  --sample-manifest "$SPLIT" \
  --backbone-checkpoint "$BASE/best.pt" \
  --trajectory-checkpoint "$RUNS/pointodyssey_full_uv_velocity_residual_60/best.pt" \
  --output-dir "$ADAPTER" \
  --adapter-init-dir "$RUNS/tau_v2_direction_rank_pb002_20261009" \
  --image-size 128 --epochs 8 --batch-size 2 --num-workers 1 --lr 0.00005 \
  --pixel-budget 0.002 --pixel-budget-weight 1.0 --direction-rank-weight 1.0 \
  --object-motion-threshold-px 4 --object-mask-radius 6 \
  > "$ADAPTER/train.log" 2>&1
test -s "$ADAPTER/summary.json"
printf 'complete\n' > "$ADAPTER/COMPLETE"
echo "$(date -Is) stage=adapter_complete"
