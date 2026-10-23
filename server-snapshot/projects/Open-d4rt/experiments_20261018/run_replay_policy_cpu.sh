set -eu
cd /root/autodl-tmp/Open-d4rt
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export ONEDNN_PRIMITIVE_CACHE_CAPACITY=0
for index in 0 1 2 3 4 5; do
  for stage in 8 12 16; do
    /root/miniconda3/envs/d4rt/bin/python -u experiments_20261018/run_replay_policy_cpu.py --case "$index" --stage "$stage"
  done
done
/root/miniconda3/envs/d4rt/bin/python -u experiments_20261018/run_replay_policy_cpu.py --evaluate
