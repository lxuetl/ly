#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/Open-d4rt
ROOT=/autodl-fs/data/d4rt_training/decisive_abc_20260828
TRAIN=/autodl-fs/data/d4rt_training
PY=/root/miniconda3/envs/d4rt/bin/python
A="$TRAIN/simvp_fair_learning_curve_20260828/fraction100/best.pt"
V3="$TRAIN/capsr_v3_object_20260825/visibility000/best.pt"
COMMON=(--cache-root data/d4rt_forecasting_cache --po-cache /autodl-fs/data/d4rt_datasets/d4rt_forecasting_cache_po50 --po-split-manifest /autodl-fs/data/d4rt_datasets/worldtrack_pointodyssey50_v2/po50/split_manifest.json --seed 42 --device cuda)
mkdir -p "$ROOT/B_composite_simvp" "$ROOT/C_pretrain_on_A"

# B and geometry pretraining run concurrently from the same strong A initialization.
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 "$PY" forecasting/train_simvp_composite_control.py \
  "${COMMON[@]}" --init-checkpoint "$A" --output-dir "$ROOT/B_composite_simvp" \
  --epochs 12 --batch-size 4 --lr 1e-4 >"$ROOT/B_composite_simvp/run.log" 2>&1 & pb=$!
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 "$PY" forecasting/train_capsr_calibrated.py \
  --mode structured "${COMMON[@]}" --trajectory-checkpoint "$TRAIN/pointodyssey50_uv_velocity_residual_60/best.pt" \
  --backbone-checkpoint "$A" --init-checkpoint "$V3" --output-dir "$ROOT/C_pretrain_on_A" \
  --epochs 12 --batch-size 4 --lr 3e-4 --dynamic-weight 3 --visibility-weight 0 --temporal-weight .10 \
  --lpips-weight .03 --gradient-weight .05 --static-identity-weight .10 --correction-weight .005 \
  --mask-budget-weight .005 --object-routing-weight .05 --calibration-budget-weight .002 \
  --calibration-temporal-weight .01 --motion-threshold .05 --num-motion-clusters 4 --sigma .035 --hidden 48 \
  >"$ROOT/C_pretrain_on_A/run.log" 2>&1 & pc=$!
wait "$pb" "$pc"

# Rebase the pretrained geometry optimizer/model onto B, reset model selection, then fine-tune.
mkdir -p "$ROOT/C_d4rt_on_B"
"$PY" -c "import torch; c=torch.load('$ROOT/C_pretrain_on_A/best.pt',map_location='cpu'); c['epoch']=0; c['best']=float('inf'); torch.save(c,'$ROOT/C_d4rt_on_B/last.pt')"
cp "$ROOT/C_pretrain_on_A/best.pt" "$ROOT/C_d4rt_on_B/best.pt"
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 "$PY" forecasting/train_capsr_calibrated.py \
  --mode structured "${COMMON[@]}" --trajectory-checkpoint "$TRAIN/pointodyssey50_uv_velocity_residual_60/best.pt" \
  --backbone-checkpoint "$ROOT/B_composite_simvp/best.pt" --init-checkpoint "$V3" --output-dir "$ROOT/C_d4rt_on_B" \
  --resume --epochs 8 --batch-size 4 --lr 2e-4 --dynamic-weight 3 --visibility-weight 0 --temporal-weight .10 \
  --lpips-weight .03 --gradient-weight .05 --static-identity-weight .10 --correction-weight .005 \
  --mask-budget-weight .005 --object-routing-weight .05 --calibration-budget-weight .002 \
  --calibration-temporal-weight .01 --motion-threshold .05 --num-motion-clusters 4 --sigma .035 --hidden 48 \
  >"$ROOT/C_d4rt_on_B/run.log" 2>&1
"$PY" forecasting/aggregate_decisive_abc.py >"$ROOT/decision.log" 2>&1
date -Is >"$ROOT/COMPLETE"
