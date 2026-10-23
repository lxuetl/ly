#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/Open-d4rt

PY=/root/miniconda3/envs/d4rt/bin/python
TRAIN=/autodl-fs/data/d4rt_training
OUTPUT=${OUTPUT:-$TRAIN/risk_calibrated_abstention_v2}

mkdir -p "$OUTPUT"

nohup env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  "$PY" forecasting/train_risk_calibrated_abstention.py \
  --cache-root data/d4rt_forecasting_cache \
  --po-cache /autodl-fs/data/d4rt_datasets/d4rt_forecasting_cache_po50 \
  --po-split-manifest /autodl-fs/data/d4rt_datasets/worldtrack_pointodyssey50_v2/po50/split_manifest.json \
  --trajectory-checkpoint "$TRAIN/pointodyssey50_uv_velocity_residual_60/best.pt" \
  --backbone-checkpoint "$TRAIN/decisive_abc_20260828/B_composite_simvp/best.pt" \
  --correction-checkpoint "$TRAIN/decisive_abc_20260828/C_d4rt_on_B/best.pt" \
  --output-dir "$OUTPUT" \
  --epochs "${EPOCHS:-16}" \
  --batch-size "${BATCH_SIZE:-2}" \
  --max-train "${MAX_TRAIN:-192}" \
  --max-val "${MAX_VAL:-48}" \
  --max-test "${MAX_TEST:-96}" \
  --lr "${LR:-4e-4}" \
  --seed 42 --device cuda \
  >"$OUTPUT/run.log" 2>&1 &

echo $! >"$OUTPUT/launcher.pid"
echo "Started PID $(cat "$OUTPUT/launcher.pid"); log: $OUTPUT/run.log"
