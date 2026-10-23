set -eu
cd /root/autodl-tmp/Open-d4rt
mkdir -p /root/autodl-fs/d4rt_training/cpu_first_query_cross_objective_20261019
nohup bash experiments_20261018/run_cross_objective_cpu.sh > /root/autodl-fs/d4rt_training/cpu_first_query_cross_objective_20261019/worker.log 2>&1 < /dev/null &
echo "Bounded cross-objective CPU diagnostic PID: $!"
