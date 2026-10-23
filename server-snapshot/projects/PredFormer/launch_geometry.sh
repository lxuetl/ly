#!/bin/bash
set -eu
cd /root/autodl-tmp/PredFormer
PY=/root/autodl-tmp/envs/predformer/bin/python
"$PY" -u geometry_experiment.py --prepare
for arm in rgb null omega d4rt; do
  nohup "$PY" -u geometry_experiment.py --arm "$arm" > "/root/autodl-tmp/predformer_geo_${arm}.log" 2>&1 < /dev/null &
  echo "$arm PID=$!"
done
wait
