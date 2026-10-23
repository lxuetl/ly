#!/usr/bin/env bash
set -euo pipefail

REPO=/root/autodl-tmp/Open-d4rt
PY=/root/miniconda3/envs/d4rt/bin/python
CACHE=/autodl-fs/data/d4rt_datasets/d4rt_forecasting_cache_po_full
SPLIT=/autodl-fs/data/d4rt_datasets/worldtrack_pointodyssey_full_v2/po50/split_manifest.json
TRAJ_OUT=/autodl-fs/data/d4rt_training/pointodyssey_full_uv_velocity_residual_60
ROUTER_OUT=/autodl-fs/data/d4rt_training/capsr_stable_router_po_full_20260919

cd "$REPO"

echo "STAGE trajectory_start $(date -Is)"
"$PY" forecasting/train_pointodyssey_joint.py \
  --cache-root data/d4rt_forecasting_cache \
  --po-cache "$CACHE" \
  --po-split-manifest "$SPLIT" \
  --output-dir "$TRAJ_OUT" \
  --epochs 60 \
  --batch-size 2 \
  --seed 42 \
  --device cuda \
  --legacy-subsets adt_mini,ds_mini,po_mini,pstudio_mini \
  --model-variant velocity_residual \
  --init-checkpoint /autodl-fs/data/d4rt_training/pointodyssey50_uv_velocity_residual_60/best.pt
echo "STAGE trajectory_done $(date -Is)"

echo "STAGE router_start $(date -Is)"
"$PY" forecasting/train_capsr_stable_router.py \
  --mode structured \
  --cache-root data/d4rt_forecasting_cache \
  --po-cache "$CACHE" \
  --po-split-manifest "$SPLIT" \
  --trajectory-checkpoint "$TRAJ_OUT/best.pt" \
  --backbone-checkpoint outputs/simvp_dynamic_ablation/no_geometry/best.pt \
  --output-dir "$ROUTER_OUT" \
  --init-checkpoint /autodl-fs/data/d4rt_training/capsr_v3_object_20260825/visibility000/best.pt \
  --freeze-legacy \
  --epochs 10 \
  --batch-size 4 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --dynamic-weight 3.0 \
  --visibility-weight 0.0 \
  --temporal-weight 0.1 \
  --lpips-weight 0.03 \
  --gradient-weight 0.05 \
  --static-identity-weight 0.1 \
  --correction-weight 0.005 \
  --mask-budget-weight 0.002 \
  --object-routing-weight 0.0 \
  --router-oracle-weight 0.01 \
  --router-regret-weight 5.0 \
  --router-temporal-weight 0.03 \
  --router-spatial-weight 0.02 \
  --router-prior-kl-weight 0.03 \
  --router-region-kernel 5 \
  --router-benefit-margin 0.0005 \
  --router-structure-cost-weight 0.1 \
  --router-candidate-temporal-cost-weight 0.1 \
  --router-low-margin-temperature 0.001 \
  --motion-threshold 0.05 \
  --num-motion-clusters 4 \
  --sigma 0.035 \
  --hidden 48 \
  --max-train 0 \
  --max-val 0 \
  --max-test 0 \
  --legacy-subsets adt_mini,ds_mini,po_mini,pstudio_mini \
  --seed 42 \
  --device cuda
echo "STAGE router_done $(date -Is)"

echo "FULL_PIPELINE_DONE $(date -Is)"
