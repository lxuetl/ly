#!/bin/bash
set -eu
cd /root/autodl-tmp/PredFormer
test ! -e /root/autodl-tmp/predformer_local_horizon_20261022
PY=/root/autodl-tmp/envs/predformer/bin/python
for arm in omega rgb null omega_shuffled; do
 nohup "$PY" -u local_horizon.py --arm "$arm" > "/root/autodl-tmp/predformer_local_${arm}.log" 2>&1 < /dev/null &
 echo "$arm PID=$!"
done
wait
