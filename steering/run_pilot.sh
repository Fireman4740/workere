#!/usr/bin/env bash
# Steering pilot on the local Qwen3.8-27B (RTX 3090), run inside the HyperQwen container.
# Stages: smoke (hooks + batched steering) -> directions -> alpha sweep at one layer -> score.
# Directions and generations persist on the box under $STEER_DATA/out, outside the run dir.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-/app/venv/bin/python}
STEER_DATA=${STEER_DATA:-/app/local_extensions/steering}
PROMPTS=${PROMPTS:-$STEER_DATA/prompts.jsonl}
DIRS=${DIRS:-$STEER_DATA/out/dirs}
RUN_OUT=$STEER_DATA/out/runs/${ORX_RUN_ID:-manual}
LAYERS=${LAYERS:-24,32,40,48}
LAYER=${LAYER:-32}
ALPHAS=${ALPHAS:-0.1,0.05,0.2}
FAMILIES=${FAMILIES:-hint,demo}
DOC_IDS=${DOC_IDS:-$REPO/steering/data/docs10.txt}
BATCH=${BATCH:-10}
export PYTHONPATH=$REPO:$REPO/steering
export VLLM_USE_FLASHINFER_SAMPLER=0 FLASHINFER_DISABLE_VERSION_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
mkdir -p "$RUN_OUT" "$DIRS"
cd "$REPO/steering"

echo "== config: layer=$LAYER alphas=$ALPHAS families=$FAMILIES batch=$BATCH docs=$(wc -l < "$DOC_IDS") commit=${ORX_COMMIT:-?}"
echo "== output: $RUN_OUT"

if [ -f "$STEER_DATA/extract_all.sh" ] && [ ! -f "$STEER_DATA/out/extract.done" ]; then
  while [ ! -f "$STEER_DATA/out/extract.done" ]; do
    echo "== waiting for the running direction extraction to free the GPU ($(date +%T))"
    sleep 60
  done
  sleep 30
fi

echo "== stage 0: smoke (hooks, graph decode, batched steering)"
$PY smoke.py --layer "$LAYER" --batch 4

echo "== stage 1: directions"
for family in ${FAMILIES//,/ }; do
  if [ -f "$DIRS/${family}_L${LAYER}_down_proj.npz" ]; then
    echo "   $family: reuse $DIRS/${family}_L${LAYER}_down_proj.npz"
  else
    $PY extract_direction.py --method "$family" --prompts "$PROMPTS" --out-dir "$DIRS" --layers "$LAYERS"
  fi
done
$PY - "$DIRS" "$LAYER" "$FAMILIES" <<'EOF'
import sys, numpy as np
dirs, layer, families = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
for family in families:
    data = np.load(f"{dirs}/{family}_L{layer}_down_proj.npz")
    diffs = data["diffs"]
    v = data["v"]
    proj = diffs @ v
    print(f"   {family} L{layer}: ref_norm={float(data['ref_norm']):.2f} "
          f"natural shift/ref_norm={float(np.linalg.norm(diffs.mean(0))) / float(data['ref_norm']):.4f} "
          f"separation_d={proj.mean() / (proj.std() + 1e-8):.2f} docs={len(diffs)}")
EOF

echo "== stage 2: alpha sweep"
$PY sweep_conditions.py --direction-dir "$DIRS" --layer "$LAYER" --alphas "$ALPHAS" \
  --families "$FAMILIES" --out "$RUN_OUT/conditions.json"
SCORE="$PY score.py --reference data/reference.json --predictions $RUN_OUT/gen.jsonl --out $RUN_OUT/scores.json --csv $RUN_OUT/scores.csv | grep -E '^[a-z]|run_docs'; $PY compare.py --csv $RUN_OUT/scores.csv --subset run_docs --draws 5000"
$PY run_vllm.py --prompts "$PROMPTS" --conditions "$RUN_OUT/conditions.json" \
  --doc-ids "$DOC_IDS" --batch "$BATCH" --out "$RUN_OUT/gen.jsonl" --after-condition "$SCORE"
echo "== done: $RUN_OUT"
