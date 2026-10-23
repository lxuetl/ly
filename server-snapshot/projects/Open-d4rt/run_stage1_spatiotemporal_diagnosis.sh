#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/Open-d4rt

ROOT=/autodl-fs/data/d4rt_training/stage1_spatiotemporal_diagnosis_20260823
mkdir -p "$ROOT"

if ! /root/miniconda3/envs/d4rt/bin/python -c 'import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)'; then
  echo "ERROR: CUDA GPU is unavailable; restore the GPU instance before running stage 1." >&2
  exit 2
fi

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0 \
  /root/miniconda3/envs/d4rt/bin/python forecasting/diagnose_spatiotemporal_benefit.py \
  --cache-root data/d4rt_forecasting_cache \
  --legacy-subsets adt_mini,ds_mini,po_mini,pstudio_mini \
  --po-cache /autodl-fs/data/d4rt_datasets/d4rt_forecasting_cache_po50 \
  --po-split-manifest /autodl-fs/data/d4rt_datasets/worldtrack_pointodyssey50_v2/po50/split_manifest.json \
  --trajectory-checkpoint /autodl-fs/data/d4rt_training/pointodyssey50_uv_velocity_residual_60/best.pt \
  --backbone-checkpoint outputs/dmvfn_motionloss_60/no_geometry/best.pt \
  --corrector-checkpoint /autodl-fs/data/d4rt_training/experiment_a_topk_sweep_20260821/k030/reliable_geometry_motion_comp/best.pt \
  --reference-summary /autodl-fs/data/d4rt_training/experiment_a_topk_sweep_20260821/k030/summary.json \
  --spatial-margin 0.002 --temporal-margin 0.0 --robust-temporal-margin 0.0005 \
  --output "$ROOT/diagnosis.json" --device cuda
