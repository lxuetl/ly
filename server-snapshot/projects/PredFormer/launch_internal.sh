#!/bin/bash
set -eu
cd /root/autodl-tmp/PredFormer
PY=/root/autodl-tmp/envs/predformer/bin/python
"$PY" -u internal_geometry.py --prepare
for arm in plain rgb null omega; do
  nohup "$PY" -u internal_geometry.py --arm "$arm" > "/root/autodl-tmp/predformer_internal_${arm}.log" 2>&1 < /dev/null &
  echo "$arm PID=$!"
done
wait
