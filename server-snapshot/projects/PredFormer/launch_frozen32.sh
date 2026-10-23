#!/bin/bash
set -eu
cd /root/autodl-tmp/PredFormer
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/root/autodl-tmp/envs/predformer/bin/python
test -f /root/autodl-tmp/predformer_frozen32_20261022/audit.json
for arm in omega rgb null omega_shuffled; do
 nohup "$PY" -u frozen_branch_diagnostic.py --arm "$arm" > "/root/autodl-tmp/frozen32_${arm}.log" 2>&1 < /dev/null &
 echo "$arm $!"
done
wait
