set -eu
cd /root/autodl-tmp/Open-d4rt
mkdir -p /root/autodl-fs/d4rt_training/cpu_reliability_audit_20261019
nohup bash experiments_20261018/run_reliability_cpu.sh > /root/autodl-fs/d4rt_training/cpu_reliability_audit_20261019/worker.log 2>&1 < /dev/null &
echo "Bounded CPU reliability worker PID: $!"
