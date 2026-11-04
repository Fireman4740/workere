#!/usr/bin/env bash
# Run on the CC-IN2P3 login node. Starts one tmux session per experiment; each session asks
# Slurm for one interactive L40S (gpu_l40s_interactive accepts srun only, not sbatch) and
# runs run_mech.sh inside the HyperQwen Apptainer image. Every experiment waits for the S0
# smoke to pass; E3 and E5 also wait for T0's training traces, so no GPU sits idle.
# Usage: launch_l40s.sh [T0 E1 E2 E3 E4 E5]
set -euo pipefail
ROOT=${ROOT:-/pbs/throng/liris/mcarlesso/steer}
REPO=$ROOT/repo
SIF=$ROOT/hyperqwen.sif
EXPS=${*:-S0 T0 E1 E2 E4 E3 E5}
mkdir -p "$ROOT/out" "$ROOT/logs"

for exp in $EXPS; do
  wait_t0=""
  case "$exp" in
    S0) ;;
    E3|E5) wait_t0="until [ -f $ROOT/out/S0/OK ] && [ -f $ROOT/out/T0/DONE ]; do sleep 300; done;" ;;
    *) wait_t0="until [ -f $ROOT/out/S0/OK ]; do sleep 60; done;" ;;
  esac
  cmd="$wait_t0 srun --partition=gpu_l40s_interactive --qos=gpu --account=liris --gpus=1 \
    --cpus-per-task=8 --mem=96G --time=2-00:00:00 --job-name=mech-$exp \
    apptainer exec --nv \
      -B $ROOT/models:/app/models -B $ROOT/data:/app/local_extensions/steering -B $ROOT:/work \
      $SIF bash $REPO/steering/exp/run_mech.sh $exp >> $ROOT/logs/$exp.log 2>&1; \
    echo \"== srun exit \$? \$(date -Is)\" >> $ROOT/logs/$exp.log"
  if tmux has-session -t "mech-$exp" 2>/dev/null; then
    echo "mech-$exp already running; skipped"
    continue
  fi
  tmux new-session -d -s "mech-$exp" "bash -lc '$cmd'"
  echo "mech-$exp started -> $ROOT/logs/$exp.log"
done
