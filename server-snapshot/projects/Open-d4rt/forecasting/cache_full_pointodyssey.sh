#!/usr/bin/env bash
set -euo pipefail

REPO=/root/autodl-tmp/Open-d4rt
PY=/root/miniconda3/envs/d4rt/bin/python
OUT=/autodl-fs/data/d4rt_datasets/d4rt_forecasting_cache_po_full

cd "$REPO"

run_shard() {
  local input_root="$1"
  local shard_name="$2"
  "$PY" forecasting/cache_d4rt_forecasting.py \
    --model-config checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml \
    --ckpt-path checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt \
    --data-root "$input_root" \
    --subsets po50 \
    --output-dir "$OUT" \
    --context-frames 16 \
    --future-frames 8 \
    --num-query-points 256 \
    --query-chunk-size 128 \
    --alignment-keep-ratio 0.85 \
    --seed 42 \
    --device cuda
  cp "$OUT/manifest.jsonl" "$OUT/manifest_${shard_name}.jsonl"
}

run_shard /autodl-fs/data/d4rt_datasets/cache_inputs_po_50_99 po_50_99
run_shard /autodl-fs/data/d4rt_datasets/cache_inputs_po_100_130 po_100_130

find "$OUT/po50" -maxdepth 1 -name '*.npz' -printf '%f\n' | sort > "$OUT/all_cache_files.txt"
echo "CACHE_PIPELINE_DONE files=$(wc -l < "$OUT/all_cache_files.txt")"
