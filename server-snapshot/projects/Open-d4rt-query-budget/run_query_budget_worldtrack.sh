#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/d4rt/bin/python}"
SOURCE_ROOT="${SOURCE_ROOT:-/root/autodl-tmp/Open-d4rt}"

cd "$ROOT"
exec "$PYTHON" scripts/query_budget/run_worldtrack_budget_sweep.py \
  --model-config "${MODEL_CONFIG:-$SOURCE_ROOT/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml}" \
  --ckpt-path "${CKPT_PATH:-$SOURCE_ROOT/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt}" \
  --data-root "${DATA_ROOT:-$SOURCE_ROOT/data/worldtrack_release}" \
  --output-root "${OUTPUT_ROOT:-outputs/query_budget_worldtrack}" \
  --budgets "${BUDGETS:-4,8,16,32,64}" \
  --selectors "${SELECTORS:-random,fps,grid,task_aware_pose,task_aware_track}" \
  --seeds "${SEEDS:-41,42,43}" \
  --subsets "${SUBSETS:-adt_mini,ds_mini}" \
  --num-frames "${NUM_FRAMES:-32}" \
  --query-chunk-size "${QUERY_CHUNK_SIZE:-4096}" \
  ${LIMIT_SEQS:+--limit-seqs "$LIMIT_SEQS"} \
  --device "${DEVICE:-cuda}" \
  --resume
