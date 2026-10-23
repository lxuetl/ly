#!/bin/bash
set -eu
cd /root/autodl-tmp/PredFormer
PY=/root/autodl-tmp/envs/predformer/bin/python
test ! -e /root/autodl-tmp/predformer_continue_matched_20261021
test ! -e /root/autodl-tmp/predformer_residual_adapter_audit_20261021
nohup "$PY" -u continue_matched.py > /root/autodl-tmp/predformer_continue_matched.log 2>&1 < /dev/null &
echo "continue PID=$!"
nohup "$PY" -u residual_adapter_audit.py > /root/autodl-tmp/predformer_residual_adapter_audit.log 2>&1 < /dev/null &
echo "audit PID=$!"
