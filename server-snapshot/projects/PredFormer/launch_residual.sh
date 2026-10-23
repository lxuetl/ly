#!/bin/bash
set -eu
cd /root/autodl-tmp/PredFormer
PY=/root/autodl-tmp/envs/predformer/bin/python
OUT=/root/autodl-tmp/predformer_residual_v2_20261021
test ! -e "$OUT"
for arm in query null rgb; do
  nohup "$PY" -u residual_queries.py --arm "$arm" --out "$OUT" --epochs 15 > "/root/autodl-tmp/predformer_residual_${arm}.log" 2>&1 < /dev/null &
  echo "$arm PID=$!"
done
