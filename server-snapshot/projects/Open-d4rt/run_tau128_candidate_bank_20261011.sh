#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/Open-d4rt
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
OUT=/root/autodl-fs/d4rt_training/tau128_candidate_bank_20261011
mkdir -p "$OUT"
nohup /root/miniconda3/envs/d4rt/bin/python -u eval_tau128_candidate_bank.py --history-replay --output-dir "$OUT" > "$OUT/run.log" 2>&1 < /dev/null &
RUN_PID=$!
printf '%s\n' "$RUN_PID" > "$OUT/pid"
printf 'started pid=%s output=%s\n' "$RUN_PID" "$OUT"
