#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/Open-d4rt
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export THREE_STAGE_OUT=/root/autodl-tmp/three_stage_audit_gpu_20261021
PY=/root/miniconda3/envs/d4rt/bin/python
"$PY" experiments_20261018/test_three_stage_stream.py
for index in 0 1 2 3 4 5 6 7; do
  "$PY" -u experiments_20261018/three_stage_audit.py rgb --index "$index"
  "$PY" -u experiments_20261018/three_stage_audit.py case --index "$index"
done
"$PY" -u experiments_20261018/three_stage_audit.py finalize
"$PY" -u experiments_20261018/verify_three_stage.py "$THREE_STAGE_OUT"
