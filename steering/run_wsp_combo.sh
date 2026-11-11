#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-/app/venv/bin/python}
OUT=${WORK:?}/out/${WSP_GROUP:?}/${WSP_DATASET:?}
export PYTHONPATH="$REPO:$REPO/steering"
export VLLM_USE_FLASHINFER_SAMPLER=0 FLASHINFER_DISABLE_VERSION_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False OMP_NUM_THREADS=4
mkdir -p "$OUT" "$(dirname "$LOCK")"
cd "$REPO"
ARGS=(--dataset "$WSP_DATASET" --out "$OUT" --source "$WSP_SOURCE"
      --history "$WORK/out" --max-model-len "$MAX_MODEL_LEN" --batch "$BATCH"
      --gpu-memory-utilization "$GPU_UTIL" --max-hours "$WSP_MAX_HOURS")
echo "WSP_COMBO_RUN run=${ORX_RUN_ID:?} commit=${ORX_COMMIT:?} dataset=$WSP_DATASET output=$OUT"
"$PY" -u -m steering.wsp_combo preflight "${ARGS[@]}"
START=$SECONDS
# Explicit corpus order. A failed predecessor also releases the next corpus;
# no decision uses its F1 and no incomplete run is scored as complete.
if [ -n "${WSP_PREDECESSOR:-}" ]; then
  while [ ! -f "$WORK/out/$WSP_GROUP/$WSP_PREDECESSOR/released.json" ]; do
    if [ "$((SECONDS-START))" -ge "$GPU_WAIT_SECONDS" ]; then exit 75; fi
    echo "WSP_COMBO_WAIT_PREDECESSOR $WSP_PREDECESSOR elapsed=$((SECONDS-START))"
    sleep 30
  done
fi
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
  if [ "$((SECONDS-START))" -ge "$GPU_WAIT_SECONDS" ]; then exit 75; fi
  echo "WSP_COMBO_WAIT_GPU used_mib=$used elapsed=$((SECONDS-START))"
  sleep 30
done
echo "WSP_COMBO_GPU_ACQUIRED dataset=$WSP_DATASET"
set +e
"$PY" -u -m steering.wsp_combo generate "${ARGS[@]}"
STATUS=$?
set -e
# Python/vLLM has exited before releasing the queue; the next job also checks memory.
"$PY" - "$OUT/released.json" "$STATUS" <<'PY'
import json,os,sys,time
with open(sys.argv[1],'w') as stream:
    json.dump(dict(exit_code=int(sys.argv[2]),run_id=os.environ.get('ORX_RUN_ID'),time=time.time()),stream)
PY
echo "WSP_COMBO_RELEASED dataset=$WSP_DATASET exit=$STATUS"
exit "$STATUS"
