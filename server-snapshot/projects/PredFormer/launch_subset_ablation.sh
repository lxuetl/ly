#!/bin/bash
set -eu
cd /root/autodl-tmp/PredFormer
PY=/root/autodl-tmp/envs/predformer/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$PY" subset_ablation.py --prepare
for variant in reference pre no_decay; do
 for arm in omega rgb null omega_shuffled; do
  nohup "$PY" -u subset_ablation.py --variant "$variant" --arm "$arm" > "/root/autodl-tmp/subset_${variant}_${arm}.log" 2>&1 < /dev/null &
  echo "$variant $arm $!"
 done
done
wait
