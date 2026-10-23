#!/usr/bin/env bash
set -euo pipefail

REPO=/root/autodl-tmp/Open-d4rt
PY=/root/miniconda3/envs/d4rt/bin/python
OUT=/autodl-fs/data/d4rt_training/fourway_fair_eval_20260922
CACHE=/autodl-fs/data/d4rt_datasets/d4rt_forecasting_cache_po_full
SPLIT=/autodl-fs/data/d4rt_datasets/worldtrack_pointodyssey_full_v2/po50/split_manifest.json
OLD_TRAJ=/autodl-fs/data/d4rt_training/pointodyssey50_uv_velocity_residual_60/best.pt
NEW_TRAJ=/autodl-fs/data/d4rt_training/pointodyssey_full_uv_velocity_residual_60/best.pt
OLD_ROUTER=/autodl-fs/data/d4rt_training/capsr_stable_router_20260828/best.pt
NEW_ROUTER=/autodl-fs/data/d4rt_training/capsr_stable_router_po_full_20260919/best.pt
BACKBONE=outputs/simvp_dynamic_ablation/no_geometry/best.pt

cd "$REPO"
mkdir -p "$OUT"

"$PY" forecasting/evaluate_trajectory_old_new_same_test.py \
  --cache-root data/d4rt_forecasting_cache \
  --po-cache "$CACHE" --po-split-manifest "$SPLIT" \
  --old-checkpoint "$OLD_TRAJ" --new-checkpoint "$NEW_TRAJ" \
  --output-dir "$OUT/trajectory" --bootstrap 10000 --device cuda \
  > "$OUT/trajectory.log" 2>&1 &
P_TRAJ=$!

run_video() {
  local label="$1" trajectory="$2" router="$3"
  "$PY" forecasting/evaluate_capsr_fourway.py \
    --variant stable_v2 --label "$label" --checkpoint "$router" \
    --cache-root data/d4rt_forecasting_cache \
    --po-cache "$CACHE" --po-split-manifest "$SPLIT" \
    --trajectory-checkpoint "$trajectory" \
    --backbone-checkpoint "$BACKBONE" \
    --output "$OUT/${label}.json" --device cuda \
    > "$OUT/${label}.log" 2>&1 &
  LAST_PID=$!
}

run_video oldtraj_oldrouter "$OLD_TRAJ" "$OLD_ROUTER"; P_OO=$LAST_PID
run_video newtraj_oldrouter "$NEW_TRAJ" "$OLD_ROUTER"; P_NO=$LAST_PID
run_video oldtraj_newrouter "$OLD_TRAJ" "$NEW_ROUTER"; P_ON=$LAST_PID
run_video newtraj_newrouter "$NEW_TRAJ" "$NEW_ROUTER"; P_NN=$LAST_PID

printf 'trajectory=%s\nold_old=%s\nnew_old=%s\nold_new=%s\nnew_new=%s\n' \
  "$P_TRAJ" "$P_OO" "$P_NO" "$P_ON" "$P_NN" > "$OUT/pids.txt"

wait "$P_TRAJ"
wait "$P_OO"
wait "$P_NO"
wait "$P_ON"
wait "$P_NN"

"$PY" forecasting/aggregate_fourway_by_sequence.py \
  --inputs \
    "$OUT/oldtraj_oldrouter.json" \
    "$OUT/newtraj_oldrouter.json" \
    "$OUT/oldtraj_newrouter.json" \
    "$OUT/newtraj_newrouter.json" \
  --output-dir "$OUT/aggregate" --bootstrap 10000 --seed 42 \
  > "$OUT/aggregate.log" 2>&1

date -Is > "$OUT/COMPLETE"
echo "FOURWAY_COMPLETE $(date -Is)"
