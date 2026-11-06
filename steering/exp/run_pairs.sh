#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
export PYTHONPATH="$REPO:$REPO/steering"
export VLLM_USE_FLASHINFER_SAMPLER=0 FLASHINFER_DISABLE_VERSION_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
PY=${PY:-/app/venv/bin/python}
WORK=${WORK:-/app/local_extensions/steering/mech}
OUT="$WORK/out/$PAIR_VARIANT/${ORX_RUN_ID:?}"
mkdir -p "$OUT"
echo "== $PAIR_VARIANT commit=${ORX_COMMIT:-unknown} output=$OUT"
cd "$REPO"
$PY -u steering/pair_verifier.py prepare --variant "$PAIR_VARIANT" --out "$OUT"
(
  flock 9
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    if [ "$used" -lt 2048 ]; then break; fi
    echo "== GPU busy: $used MiB; waiting"
    sleep 60
  done
  echo "== GPU acquired $(date -Is)"
  $PY -u steering/pair_verifier.py infer --variant "$PAIR_VARIANT" --out "$OUT"
) 9>"${LOCK:-/app/local_extensions/steering/out/gpu.lock}"
$PY -u steering/pair_verifier.py score --variant "$PAIR_VARIANT" --out "$OUT" --work "$WORK"
echo "== $PAIR_VARIANT complete $(date -Is)"
