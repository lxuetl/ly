#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/Open-d4rt}"
PYTHON="${PYTHON:-/root/miniconda3/envs/d4rt/bin/python}"
MANIFEST="${MANIFEST:-/autodl-fs/data/d4rt_training/simvp_64_strict_fraction025_20260926/split_paths.json}"
OUT_ROOT="${OUT_ROOT:-/autodl-fs/data/d4rt_training/four_backbone_fair64_20260927}"
EPOCHS="${EPOCHS:-15}"

cd "$ROOT"
mkdir -p "$OUT_ROOT"
export OMP_NUM_THREADS=2

common=(
  --cache-root data/d4rt_forecasting_cache
  --po-cache /autodl-fs/data/d4rt_datasets/d4rt_forecasting_cache_po_full
  --po-split-manifest /autodl-fs/data/d4rt_datasets/worldtrack_pointodyssey_full_v2/po50/split_manifest.json
  --sample-manifest "$MANIFEST"
  --image-size 64
  --epochs "$EPOCHS"
  --patience 5
  --seed 42
)

declare -A batches=(
  [simvp]=2
  [tau]=2
  [dmvfn]=2
  [predrnnpp]=2
)

pids=()
for backbone in simvp tau dmvfn predrnnpp; do
  run_dir="$OUT_ROOT/$backbone"
  mkdir -p "$run_dir"
  "$PYTHON" train_four_backbone_fair.py \
    --backbone "$backbone" \
    --output-dir "$run_dir" \
    --batch-size "${batches[$backbone]}" \
    "${common[@]}" \
    >"$run_dir/train.log" 2>&1 &
  pid=$!
  pids+=("$pid")
  printf '%s\t%s\n' "$backbone" "$pid" | tee -a "$OUT_ROOT/pids.tsv"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
date -Iseconds > "$OUT_ROOT/FINISHED_AT"
exit "$status"
