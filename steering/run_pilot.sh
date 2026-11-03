#!/usr/bin/env bash
# Steering pilot on the local Qwen3.8-27B (RTX 3090), run inside the HyperQwen container.
# Stages: smoke (hooks + batched steering) -> directions -> alpha sweep at one layer -> score.
# Directions and generations persist on the box under $STEER_DATA/out, outside the run dir.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
# Per-experiment settings live in steering/config.env on each experiment branch.
if [ -f "$REPO/steering/config.env" ]; then set -a; . "$REPO/steering/config.env"; set +a; fi
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
MAX_MODEL_LEN=${MAX_MODEL_LEN:-24576}
MAX_TOKENS=${MAX_TOKENS:-20480}
GPU_UTIL=${GPU_UTIL:-0.90}
export PYTHONPATH=$REPO:$REPO/steering
export VLLM_USE_FLASHINFER_SAMPLER=0 FLASHINFER_DISABLE_VERSION_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
mkdir -p "$RUN_OUT" "$DIRS"
cd "$REPO/steering"

echo "== config: layer=$LAYER alphas=$ALPHAS families=$FAMILIES batch=$BATCH max_model_len=$MAX_MODEL_LEN max_tokens=$MAX_TOKENS gpu_util=$GPU_UTIL docs=$(wc -l < "$DOC_IDS") commit=${ORX_COMMIT:-?}"
echo "== output: $RUN_OUT"

exec 9>"$STEER_DATA/out/gpu.lock"
if ! flock -n 9; then
  echo "== another steering run holds the GPU; queued ($(date +%T))"
  flock 9
fi
# Under WSL, torch.cuda.mem_get_info does not see memory held by Windows processes.
until used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1) && [ "$used" -lt "${GPU_BUSY_MIB:-2048}" ]; do
  echo "== GPU busy: ${used} MiB used by another process; waiting ($(date +%T))"
  sleep 120
done
echo "== GPU acquired ($(date +%T))"

if [ -f "$STEER_DATA/extract_all.sh" ] && [ ! -f "$STEER_DATA/out/extract.done" ]; then
  while [ ! -f "$STEER_DATA/out/extract.done" ]; do
    echo "== waiting for the running direction extraction to free the GPU ($(date +%T))"
    sleep 60
  done
  sleep 30
fi

if [ "${SKIP_SMOKE:-0}" != 1 ]; then
  echo "== stage 0: smoke (hooks, graph decode, batched steering)"
  $PY smoke.py --layer "$LAYER" --batch 4
fi

echo "== stage 1: directions"
for family in $(echo "${FAMILIES//,/ }" | tr ' ' '\n' | sed 's/_.*//' | sort -u); do
  if [ -f "$DIRS/${family}_L${LAYER}_down_proj.npz" ]; then
    echo "   $family: reuse $DIRS/${family}_L${LAYER}_down_proj.npz"
  else
    $PY extract_direction.py --method "$family" --prompts "$PROMPTS" --out-dir "$DIRS" --layers "$LAYERS"
  fi
done
for family in ${FAMILIES//,/ }; do
  case "$family" in *_neg|*_rand*)
    base=${family%%_*}
    [ -f "$DIRS/${base}_L${LAYER}_down_proj_${family#*_}.npz" ] || $PY make_controls.py --direction "$DIRS/${base}_L${LAYER}_down_proj.npz" --random 2 ;;
  esac
done
$PY - "$DIRS" "$LAYER" "$(echo "${FAMILIES//,/ }" | tr ' ' '\n' | sed 's/_.*//' | sort -u | paste -sd, -)" <<'EOF'
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
  --doc-ids "$DOC_IDS" --batch "$BATCH" --max-model-len "$MAX_MODEL_LEN" --max-tokens "$MAX_TOKENS" \
  --gpu-memory-utilization "$GPU_UTIL" --out "$RUN_OUT/gen.jsonl" --after-condition "$SCORE"
echo "== done: $RUN_OUT"
