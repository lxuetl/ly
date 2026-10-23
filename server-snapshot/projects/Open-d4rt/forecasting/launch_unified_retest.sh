#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/Open-d4rt

OUT=/autodl-fs/data/d4rt_training/unified_retest_20260828
TRAIN=/autodl-fs/data/d4rt_training
PY=/root/miniconda3/envs/d4rt/bin/python
mkdir -p "$OUT"

COMMON=(
  --cache-root data/d4rt_forecasting_cache
  --po-cache /autodl-fs/data/d4rt_datasets/d4rt_forecasting_cache_po50
  --po-split-manifest /autodl-fs/data/d4rt_datasets/worldtrack_pointodyssey50_v2/po50/split_manifest.json
  --trajectory-checkpoint "$TRAIN/pointodyssey50_uv_velocity_residual_60/best.pt"
  --backbone-checkpoint outputs/simvp_dynamic_ablation/no_geometry/best.pt
  --epochs 0 --batch-size 4 --seed 42
  --motion-threshold 0.05 --num-motion-clusters 4
  --sigma 0.035 --hidden 48 --device cuda --resume
)

prepare_output() {
  local name=$1
  local source=$2
  mkdir -p "$OUT/$name"
  cp "$source/best.pt" "$OUT/$name/best.pt"
  cp "$source/last.pt" "$OUT/$name/last.pt"
  if [[ -f "$source/history.json" ]]; then
    cp "$source/history.json" "$OUT/$name/history.json"
  fi
}

run_eval() {
  local name=$1
  local script=$2
  local mode=$3
  shift 3
  echo "$(date -Is) START $name" | tee -a "$OUT/status.log"
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    "$PY" "$script" --mode "$mode" --output-dir "$OUT/$name" \
    "${COMMON[@]}" "$@" >"$OUT/$name/eval.log" 2>&1
  echo "$(date -Is) DONE $name" | tee -a "$OUT/status.log"
}

V3="$TRAIN/capsr_v3_object_20260825/visibility000"

prepare_output no_geometry "$TRAIN/capsr_v2_fair_20260825/no_geometry"
prepare_output capsr_v3 "$V3"
prepare_output capsr_v5 "$TRAIN/capsr_calibrated_v5_20260827"
prepare_output capsr_v6 "$TRAIN/capsr_constrained_v6_20260827"
prepare_output router_v1 "$TRAIN/capsr_multipath_router_20260827"
prepare_output stable_router_v2 "$TRAIN/capsr_stable_router_20260828"

# Wave 1: three models sharing one 32 GB GPU.
run_eval no_geometry forecasting/train_capsr_structured.py no_geometry & p1=$!
run_eval capsr_v3 forecasting/train_capsr_structured.py structured & p2=$!
run_eval capsr_v5 forecasting/train_capsr_calibrated.py structured \
  --init-checkpoint "$V3/best.pt" --freeze-legacy & p3=$!
wait "$p1" "$p2" "$p3"

# Wave 2: the constrained and routing variants.
run_eval capsr_v6 forecasting/train_capsr_constrained_v6.py structured \
  --init-checkpoint "$V3/best.pt" --freeze-legacy & p4=$!
run_eval router_v1 forecasting/train_capsr_multipath_router.py structured \
  --init-checkpoint "$V3/best.pt" --freeze-legacy & p5=$!
run_eval stable_router_v2 forecasting/train_capsr_stable_router.py structured \
  --init-checkpoint "$V3/best.pt" --freeze-legacy & p6=$!
wait "$p4" "$p5" "$p6"

date -Is >"$OUT/COMPLETE"
echo "$(date -Is) ALL_DONE" | tee -a "$OUT/status.log"
