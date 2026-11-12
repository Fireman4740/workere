#!/usr/bin/env bash
# Run on the CC-IN2P3 login node after `maven_pool prepare`. Starts one interactive L40S
# allocation per shard (gpu_l40s_interactive accepts srun only), detached with setsid/nohup
# because the login alias round-robins between hosts. Resumable: rerun to finish pending reads.
# Usage: launch_maven_pool_l40s.sh [shards=4]
set -euo pipefail
SHARDS=${1:-4}
ROOT=${ROOT:-/sps/liris/mcarlesso/maven-pool}
mkdir -p "$ROOT/logs" "$ROOT/bin"
# bash reads a script while it runs: give the jobs a frozen copy so a later sync cannot corrupt them
SCRIPT=$ROOT/bin/maven_pool_shard-$(date +%Y%m%d-%H%M%S).sh
cp "$ROOT/repo/steering/exp/maven_pool_shard.sh" "$SCRIPT"
for ((s = 0; s < SHARDS; s++)); do
  log=$ROOT/logs/srun-$s-$(date +%Y%m%d-%H%M%S).log
  setsid nohup srun --partition=gpu_l40s_interactive --qos=gpu --account=liris --gpus=1 \
    --cpus-per-task=8 --mem=96G --time=1-00:00:00 --licenses=sps --job-name=mp-$s \
    bash "$SCRIPT" "$s" "$SHARDS" > "$log" 2>&1 < /dev/null &
  echo "shard $s/$SHARDS -> $log"
done
