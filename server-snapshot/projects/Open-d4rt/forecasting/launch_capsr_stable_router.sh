#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/Open-d4rt
root_dir=/autodl-fs/data/d4rt_training/capsr_stable_router_20260828
mkdir -p "$root_dir"
if [[ -f "$root_dir/pid" ]] && kill -0 "$(<"$root_dir/pid")" 2>/dev/null; then
  echo "already running pid=$(<"$root_dir/pid")"
  exit 0
fi
date +%s >"$root_dir/start_epoch_seconds"
nohup bash forecasting/run_capsr_stable_router.sh >"$root_dir/launcher.log" 2>&1 &
pid=$!
echo "$pid" >"$root_dir/pid"
echo "started pid=$pid"
