#!/usr/bin/env bash
# Fixed parent launch command dispatches here. Never stop/restart another service.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-/app/venv/bin/python}
OUT=${WORK:-/app/local_extensions/steering/mech}/out/${WSP_GROUP:?}
export PYTHONPATH="$REPO:$REPO/steering"
export VLLM_USE_FLASHINFER_SAMPLER=0 FLASHINFER_DISABLE_VERSION_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export OMP_NUM_THREADS=4
mkdir -p "$OUT" "$(dirname "$LOCK")"
cd "$REPO"
ARGS=(--out "$OUT" --max-model-len "$MAX_MODEL_LEN" --batch "$BATCH"
      --gpu-memory-utilization "$GPU_UTIL")
MODULE=steering.wsp_pilot
if [ "${WSP_MODE:-screen}" = confirmation ]; then
  MODULE=steering.wsp_confirmation
  ARGS+=(--source-dir "${WSP_SOURCE:?}")
elif [ "${WSP_MODE:-screen}" != screen ]; then
  echo "Unknown WSP_MODE: $WSP_MODE" >&2
  exit 2
fi
echo "WSP_RUN run=${ORX_RUN_ID:?} commit=${ORX_COMMIT:?} output=$OUT"
"$PY" -u -m "$MODULE" prepare "${ARGS[@]}"

# Busy server on the Windows side can be invisible in this container's process list.
# Wait for memory to be free AND take the existing shared lock. Never hold the lock
# while waiting for GPU memory; do not block another queued experiment behind us.
START=$SECONDS
NOTIFIED=0
while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [[ "$used" =~ ^[0-9]+$ ]] && [ "$used" -lt "$GPU_BUSY_MIB" ]; then
    exec 9>"$LOCK"
    if flock -n 9; then
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
      if [[ "$used" =~ ^[0-9]+$ ]] && [ "$used" -lt "$GPU_BUSY_MIB" ]; then break; fi
      flock -u 9
    fi
    exec 9>&-
  fi
  echo "WSP_WAIT_GPU used_mib=$used elapsed_seconds=$((SECONDS-START)); shared server remains running"
  if [ "$NOTIFIED" = 0 ] && [ -n "${ORX_NOTIFY_FILE:-}" ]; then
    echo "WSP prêt : audit des données terminé, en attente de la 3090 occupée par le serveur partagé. Aucun autre calcul interrompu." >> "$ORX_NOTIFY_FILE"
    NOTIFIED=1
  fi
  if [ "$((SECONDS-START))" -ge "$GPU_WAIT_SECONDS" ]; then
    echo 'WSP_BLOCKED_GPU: wait limit reached; no model evaluation performed' >&2
    exit 75
  fi
  sleep 30
done
echo "WSP_GPU_ACQUIRED elapsed_seconds=$((SECONDS-START))"
if [ "$MODULE" = steering.wsp_pilot ]; then
  "$PY" -u -m "$MODULE" extract "${ARGS[@]}"
fi
"$PY" -u -m "$MODULE" generate "${ARGS[@]}"
"$PY" -u -m "$MODULE" score "${ARGS[@]}"
echo "WSP_DONE output=$OUT"
