#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
export PYTHONPATH="$REPO:$REPO/steering"
export VLLM_USE_FLASHINFER_SAMPLER=0 FLASHINFER_DISABLE_VERSION_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_DISABLE_COMPILE_CACHE=1
PY=${PY:-/app/venv/bin/python}
WORK=${WORK:-/app/local_extensions/steering/mech}
OUT="$WORK/out/C3/${ORX_RUN_ID:?}"
SOURCE="$WORK/out/C2/${C2_RUN:?}"
mkdir -p "$OUT"
cd "$REPO"
echo "== C3 commit=${ORX_COMMIT:-unknown} source=$SOURCE output=$OUT $(date -Is)"
deadline=$((SECONDS + 21600))
until [ -s "$SOURCE/report.json" ]; do
  if [ "$SECONDS" -ge "$deadline" ]; then
    echo 'ERROR: C2 final report unavailable after six hours; no inference launched' >&2
    exit 1
  fi
  echo "== waiting for C2 final report $(date -Is)"
  sleep 60
done
$PY -u steering/campaign_adjudicate.py prepare --out "$OUT" --source "$SOURCE" --work "$WORK"
if [ -n "${RESUME_RUN:-}" ]; then
  $PY -u steering/campaign_adjudicate.py resume --out "$OUT" --source "$WORK/out/C3/$RESUME_RUN"
fi
if ! $PY -u steering/campaign_adjudicate.py complete --out "$OUT"; then
  (
    flock 9
    while true; do
      if ! used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); then
        echo 'ERROR: GPU inaccessible to container' >&2
        exit 1
      fi
      if [ "$used" -lt 2048 ]; then break; fi
      echo "== GPU busy: $used MiB $(date -Is)"
      sleep 60
    done
    echo "== GPU acquired $(date -Is)"
    $PY -u steering/campaign_adjudicate.py infer --out "$OUT" --source "$SOURCE"
  ) 9>"${LOCK:-/app/local_extensions/steering/out/gpu.lock}"
fi
$PY -u steering/campaign_adjudicate.py score --out "$OUT"
echo "== C3 complete $(date -Is)"
