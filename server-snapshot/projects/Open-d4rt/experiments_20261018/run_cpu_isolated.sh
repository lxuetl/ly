set -eu
cd /root/autodl-tmp/Open-d4rt
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
for index in 0 3 4; do
  /root/miniconda3/envs/d4rt/bin/python -u experiments_20261018/run_cpu_history_endpoint_isolated.py --case "$index"
done
/root/miniconda3/envs/d4rt/bin/python -u experiments_20261018/run_cpu_history_endpoint_isolated.py --finalize
