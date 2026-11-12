#!/usr/bin/env bash
# Run on the CC-IN2P3 login node after `maven_pool prepare`. gpu_l40s_interactive accepts srun
# only, and a srun left running on the login node is killed when the last ssh session on that
# host closes (seen 2026-11-12: all shards "CANCELLED by" our own uid 4 min after launch).
# So each shard gets a small htc_daemon batch job whose only task is that srun: the batch job
# survives logouts, and the srun inside it holds one L40S. Resumable: rerun to finish pending reads.
# Usage: launch_maven_pool_l40s.sh [shards=4]
set -euo pipefail
SHARDS=${1:-4}
ROOT=${ROOT:-/sps/liris/mcarlesso/maven-pool}
STAMP=$(date +%Y%m%d-%H%M%S)
mkdir -p "$ROOT/logs" "$ROOT/bin"
# bash reads a script while it runs: give the jobs a frozen copy so a later sync cannot corrupt them
SCRIPT=$ROOT/bin/maven_pool_shard-$STAMP.sh
cp "$ROOT/repo/steering/exp/maven_pool_shard.sh" "$SCRIPT"
WRAP=$ROOT/bin/l40s_wrap-$STAMP.sh
cat > "$WRAP" <<'EOF'
#!/bin/bash
# Inside a batch job srun would add a step to this job; drop SLURM_* to request a new L40S job.
for v in $(env | grep -o '^SLURM_[A-Z_]*'); do unset "$v"; done
srun --partition=gpu_l40s_interactive --qos=gpu --account=liris --gpus=1 --ntasks=1 \
  --cpus-per-task=8 --mem=96G --time=1-00:00:00 --licenses=sps --job-name="mp-$2" \
  bash "$1" "$2" "$3"
echo "srun rc=$? $(date -Is)"
EOF
for ((s = 0; s < SHARDS; s++)); do
  log=$ROOT/logs/srun-$s-$STAMP.log
  sbatch --parsable -p htc_daemon -A liris -c 1 --mem=2G -t 1-00:00:00 -J "mpw-$s" -o "$log" \
    "$WRAP" "$SCRIPT" "$s" "$SHARDS"
  echo "shard $s/$SHARDS -> $log"
done
