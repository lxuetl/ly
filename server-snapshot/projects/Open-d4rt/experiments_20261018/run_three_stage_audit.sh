#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/Open-d4rt
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
PY=/root/miniconda3/envs/d4rt/bin/python
if [ ! -f /root/autodl-tmp/three_stage_audit_20261020/geometry_summary.json ]; then
  "$PY" -u experiments_20261018/three_stage_audit.py geometry
fi
for index in 0 1 2 3 4 5 6 7; do
  if [ ! -f "/root/autodl-tmp/three_stage_audit_20261020/case_${index}.json" ]; then
    "$PY" -u experiments_20261018/three_stage_audit.py rgb --index "$index"
  fi
  "$PY" -u experiments_20261018/three_stage_audit.py case --index "$index"
done
"$PY" -u experiments_20261018/three_stage_audit.py finalize
"$PY" -u experiments_20261018/verify_three_stage.py /root/autodl-tmp/three_stage_audit_20261020
