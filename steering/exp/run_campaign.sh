#!/usr/bin/env bash
# C1 paired factorial. All source, model outputs and scores are tracked by the run.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
export PYTHONPATH="$REPO:$REPO/steering"
export VLLM_USE_FLASHINFER_SAMPLER=0 FLASHINFER_DISABLE_VERSION_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_DISABLE_COMPILE_CACHE=1
PY=${PY:-/app/venv/bin/python}
WORK=${WORK:-/app/local_extensions/steering/mech}
OUT="$WORK/out/C1/${ORX_RUN_ID:?}"
mkdir -p "$OUT"
cd "$REPO"
echo "== C1 commit=${ORX_COMMIT:-unknown} output=$OUT $(date -Is)"
$PY -u steering/campaign_blocks.py prepare --out "$OUT"
if [ -n "${RESUME_RUN:-}" ]; then
  $PY -u steering/campaign_blocks.py resume --out "$OUT" --source "$WORK/out/C1/$RESUME_RUN"
fi
if ! $PY -u steering/campaign_blocks.py complete --out "$OUT"; then
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
    $PY -u steering/campaign_blocks.py infer --out "$OUT"
  ) 9>"${LOCK:-/app/local_extensions/steering/out/gpu.lock}"
fi
$PY -u steering/campaign_blocks.py score --out "$OUT" --work "$WORK"
echo "== C1 complete $(date -Is)"
