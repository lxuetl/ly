#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0

ROOT=/root/autodl-tmp/Open-d4rt
TRAIN_LABELS=/autodl-fs/data/d4rt_training/vggt_critic_teacher_train216_20260924
VAL_LABELS=/autodl-fs/data/d4rt_training/vggt_critic_teacher_val80_20260924
TEST_LABELS=/autodl-fs/data/d4rt_training/vggt_geometry_critic_validation_20260924
OUTPUT=/autodl-fs/data/d4rt_training/dual_geometry_critic_adapter_20260924

mkdir -p "$OUTPUT"
while [[ ! -f "$VAL_LABELS/COMPLETE" ]]; do
  sleep 30
done

cd "$ROOT"
/root/miniconda3/envs/d4rt/bin/python train_dual_geometry_critic_adapter.py \
  --d4rt-root "$ROOT" \
  --cache-root data/d4rt_forecasting_cache \
  --po-cache /autodl-fs/data/d4rt_datasets/d4rt_forecasting_cache_po_full \
  --po-split-manifest /autodl-fs/data/d4rt_datasets/worldtrack_pointodyssey_full_v2/po50/split_manifest.json \
  --geometry-cache /autodl-fs/data/d4rt_datasets/vggt_omega_geometry_cache_512_20260924 \
  --backbone-checkpoint /autodl-fs/data/d4rt_training/simvp_fair_po_full_20260922/best.pt \
  --old-geometry-checkpoint /autodl-fs/data/d4rt_training/vggt_omega_injection_20260924/vggt_omega_geometry_best.pt \
  --old-control-checkpoint /autodl-fs/data/d4rt_training/vggt_omega_injection_20260924/rgb_capacity_control_best.pt \
  --teacher-train-jsonl "$TRAIN_LABELS/per_sequence.jsonl" \
  --teacher-val-jsonl "$VAL_LABELS/per_sequence.jsonl" \
  --teacher-test-jsonl "$TEST_LABELS/per_sequence.jsonl" \
  --output-dir "$OUTPUT" \
  --critic-epochs 12 \
  --adapter-epochs 12 \
  --batch-size 2
