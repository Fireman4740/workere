#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
export PYTHONPATH="$REPO:$REPO/steering"
export VLLM_USE_FLASHINFER_SAMPLER=0 FLASHINFER_DISABLE_VERSION_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_DISABLE_COMPILE_CACHE=1
PY=${PY:-/app/venv/bin/python}
WORK=${WORK:-/app/local_extensions/steering/mech}
OUT="$WORK/out/C2/${ORX_RUN_ID:?}"
SOURCE="$WORK/out/C1/${C1_RUN:?}"
mkdir -p "$OUT"
cd "$REPO"
echo "== C2 commit=${ORX_COMMIT:-unknown} source=$SOURCE output=$OUT $(date -Is)"
$PY -u steering/campaign_facts.py prepare --out "$OUT" --source "$SOURCE"
if [ -n "${RESUME_RUN:-}" ]; then
  $PY -u steering/campaign_facts.py resume --out "$OUT" --source "$WORK/out/C2/$RESUME_RUN"
fi
if ! $PY -u steering/campaign_facts.py complete --out "$OUT"; then
  (
    flock 9
    while true; do
      if ! used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); then
        echo 'ERROR: GPU is not accessible to the container' >&2
        exit 1
      fi
      if [ "$used" -lt 2048 ]; then break; fi
      echo "== GPU busy: $used MiB; waiting $(date -Is)"
      sleep 60
    done
    echo "== GPU acquired $(date -Is)"
    $PY -u steering/campaign_facts.py infer --out "$OUT" --source "$SOURCE"
  ) 9>"${LOCK:-/app/local_extensions/steering/out/gpu.lock}"
fi
$PY -u steering/campaign_facts.py score --out "$OUT" --source "$SOURCE" --work "$WORK"
echo "== C2 complete $(date -Is)"
