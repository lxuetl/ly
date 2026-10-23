#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/Open-d4rt

ROOT=/autodl-fs/data/d4rt_training/oracle_ladder_simvp_20260824
mkdir -p "$ROOT"

COMMON=(
  --cache-root data/d4rt_forecasting_cache
  --po-cache /autodl-fs/data/d4rt_datasets/d4rt_forecasting_cache_po50
  --po-split-manifest /autodl-fs/data/d4rt_datasets/worldtrack_pointodyssey50_v2/po50/split_manifest.json
  --trajectory-checkpoint /autodl-fs/data/d4rt_training/pointodyssey50_uv_velocity_residual_60/best.pt
  --backbone-checkpoint outputs/simvp_dynamic_ablation/no_geometry/best.pt
  --corrector-checkpoint /autodl-fs/data/d4rt_training/backbone_v7_parallel_20260821/simvp/reliable_geometry_motion_comp/best.pt
  --reference-summary /autodl-fs/data/d4rt_training/backbone_v7_parallel_20260821/simvp/summary.json
  --device cuda
)

MODES=(
  no_geometry
  history_uv
  predicted_uv
  oracle_uv
  oracle_uv_visibility
  oracle_xyz_uv_visibility
)

pids=()
for mode in "${MODES[@]}"; do
  mode_dir="$ROOT/$mode"
  mkdir -p "$mode_dir"
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    /root/miniconda3/envs/d4rt/bin/python forecasting/evaluate_oracle_ladder.py \
      "${COMMON[@]}" --mode "$mode" --output-dir "$mode_dir" \
      >"$mode_dir/run.log" 2>&1 &
  pid=$!
  pids+=("$pid")
  echo "$pid" >"$mode_dir/run.pid"
  echo "launched mode=$mode pid=$pid"
done

failed=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "completed mode=${MODES[$i]} pid=${pids[$i]}"
  else
    status=$?
    echo "failed mode=${MODES[$i]} pid=${pids[$i]} status=$status"
    failed=1
  fi
done

/root/miniconda3/envs/d4rt/bin/python - <<'PY'
import json
from pathlib import Path

root = Path('/autodl-fs/data/d4rt_training/oracle_ladder_simvp_20260824')
modes = (
    'no_geometry', 'history_uv', 'predicted_uv', 'oracle_uv',
    'oracle_uv_visibility', 'oracle_xyz_uv_visibility',
)
rows = {
    mode: json.loads((root / mode / 'result.json').read_text())
    for mode in modes
}
(root / 'summary.json').write_text(json.dumps(rows, indent=2))
print(f'wrote {root / "summary.json"} with {len(rows)} modes')
PY

exit "$failed"
