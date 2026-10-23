set -eu
cd /root/autodl-tmp/Open-d4rt
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
/root/miniconda3/envs/d4rt/bin/python -u experiments_20261018/run_first_query_cpu.py --audit
for index in 0 1 2 3; do
  /root/miniconda3/envs/d4rt/bin/python -u experiments_20261018/run_first_query_cpu.py --case "$index"
done
/root/miniconda3/envs/d4rt/bin/python -u experiments_20261018/run_first_query_cpu.py --finalize
