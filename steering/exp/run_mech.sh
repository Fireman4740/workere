#!/usr/bin/env bash
# Reasoning-mechanism experiments on Qwen3.8-27B W4A16 (same checkpoint as the 3090), one
# L40S per experiment, inside the HyperQwen image (Apptainer). Usage: run_mech.sh S0|T0|E1|...|E5
#   S0  smoke: hooks, decode graphs and batched steering on this GPU and image
#   T0  greedy traces on the 133 CTB training documents and the 47 evaluation documents
#   E1  decoding: greedy control vs temperature 0.6 vs 8k thinking budget + free answer
#   E2  early exit by agreeing free continuations (BAEE-style), paired with its own control
#   E3  projection removal of the hesitation direction (Reflection Steering), + random control
#   E4  linear pair probe on the residual stream: A before reasoning, B after own reasoning
#   E5  per-head output edit along the hesitation direction (ThinkEdit), + random heads
set -euo pipefail
EXP=$1
REPO=$(cd "$(dirname "$0")/../.." && pwd)
WORK=${WORK:-/work}
PY=${PY:-/app/venv/bin/python}
DATA=${DATA:-/app/local_extensions/steering}
OUT=$WORK/out/$EXP
EVAL_DOCS=$REPO/steering/data/docs47.txt
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
MAX_TOKENS=${MAX_TOKENS:-28672}
TRACE_TOKENS=${TRACE_TOKENS:-24576}
BATCH=${BATCH:-24}
GPU_UTIL=${GPU_UTIL:-0.92}
export PYTHONPATH=$REPO:$REPO/steering
export VLLM_USE_FLASHINFER_SAMPLER=0 FLASHINFER_DISABLE_VERSION_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export HOME=${TMPDIR:-/tmp}/qwen-home-$EXP
mkdir -p "$OUT" "$HOME"
cd "$REPO/steering"
echo "== $EXP on $(hostname) $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) commit=$(cat "$REPO/COMMIT" 2>/dev/null || echo ?) $(date -Is)"

score() {
  $PY score.py --reference data/reference.json --predictions "$1" --out "${1%.jsonl}.scores.json" \
    --csv "${1%.jsonl}.scores.csv" | grep -E '^[a-z_]|run_docs' || true
  $PY compare.py --csv "${1%.jsonl}.scores.csv" --subset run_docs --draws 10000 || true
}

generate() {  # conditions.json out.jsonl [prompts] [doc-ids] [max-tokens]
  local after="$PY score.py --reference data/reference.json --predictions $2 --out ${2%.jsonl}.scores.json --csv ${2%.jsonl}.scores.csv | grep -E '^[a-z_]|run_docs'; $PY compare.py --csv ${2%.jsonl}.scores.csv --subset run_docs --draws 10000"
  local docs_arg=()
  [ -n "${4:-}" ] && docs_arg=(--doc-ids "$4")
  gpu $PY run_vllm.py --prompts "${3:-$DATA/prompts.jsonl}" --conditions "$1" "${docs_arg[@]}" \
    --batch "$BATCH" --max-model-len "$MAX_MODEL_LEN" --max-tokens "${5:-$MAX_TOKENS}" \
    --gpu-memory-utilization "$GPU_UTIL" --out "$2" ${6:+--after-condition "$after"}
}

conditions() {  # python literal list -> json file
  $PY -c "import json,sys; json.dump($2, open(sys.argv[1], 'w'), indent=1)" "$1"
}

# On a shared GPU (the 3090) set LOCK: each GPU step then holds the lock only while it runs,
# so a job waiting for T0 never blocks the GPU, and memory held by other processes (Windows
# side under WSL, invisible to torch) is waited out first.
gpu() {
  if [ -z "${LOCK:-}" ]; then "$@"; return; fi
  (
    flock 9
    until used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1) \
        && [ "$used" -lt "${GPU_BUSY_MIB:-2048}" ]; do
      echo "== GPU busy: ${used} MiB used by another process; waiting ($(date +%T))"
      sleep 120
    done
    echo "== GPU acquired ($(date +%T))"
    "$@"
  ) 9>"$LOCK"
}

case "$EXP" in
S0)
  gpu $PY smoke.py --layer 32 --batch 4
  touch "$OUT/OK"
  ;;
T0)
  $PY ctb.py --out "$OUT/train_prompts.jsonl"
  conditions "$OUT/greedy.json" '[{"name": "control", "direction": None, "alpha": 0.0, "layer": 40, "module": "down_proj"}]'
  generate "$OUT/greedy.json" "$OUT/train_traces.jsonl" "$OUT/train_prompts.jsonl" "" "$TRACE_TOKENS"
  generate "$OUT/greedy.json" "$OUT/eval_traces.jsonl" "$DATA/prompts.jsonl" "$EVAL_DOCS" "$TRACE_TOKENS"
  touch "$OUT/DONE"
  ;;
E1)
  conditions "$OUT/conditions.json" '[
    {"name": "control", "direction": None, "alpha": 0.0, "layer": 40, "module": "down_proj"},
    {"name": "temp0.6", "temperature": 0.6, "seed": 1, "direction": None, "alpha": 0.0, "layer": 40, "module": "down_proj"},
    {"name": "budget8k", "budget": 8192, "answer_tokens": 4096, "direction": None, "alpha": 0.0, "layer": 40, "module": "down_proj"}]'
  generate "$OUT/conditions.json" "$OUT/gen.jsonl" "$DATA/prompts.jsonl" "$EVAL_DOCS" "$MAX_TOKENS" live
  ;;
E2)
  gpu $PY early_exit.py --prompts "$DATA/prompts.jsonl" --doc-ids "$EVAL_DOCS" --out "$OUT/gen.jsonl" \
    --max-tokens "${EXIT_TOKENS:-26000}" --batch "$BATCH" --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_UTIL"
  score "$OUT/gen.jsonl"
  $PY - "$OUT/gen.jsonl" <<'EOF'
import json, sys, statistics as st
rows = [json.loads(l) for l in open(sys.argv[1])]
ctrl = {r["doc_id"]: r["num_tokens"] for r in rows if r["condition"] == "control"}
early = [r for r in rows if r["condition"] == "early_exit"]
saved = [1 - r["num_tokens"] / ctrl[r["doc_id"]] for r in early if r.get("exit") == "agreement" and ctrl.get(r["doc_id"])]
print(f"early decisions by agreement: {len(saved)}/{len(early)}; median serial tokens saved {100 * st.median(saved) if saved else 0:.1f}%")
EOF
  ;;
E3)
  if [ "${RUN_T0_FIRST:-0}" = 1 ] && [ ! -f "$WORK/out/T0/DONE" ]; then bash "$0" T0; fi
  until [ -f "$WORK/out/T0/DONE" ]; do echo "== waiting for T0 traces ($(date +%T))"; sleep 300; done
  gpu $PY reflection.py --stage e3 --prompts "$WORK/out/T0/train_prompts.jsonl" --traces "$WORK/out/T0/train_traces.jsonl" \
    --out-dir "$OUT/dirs" --max-model-len "$MAX_MODEL_LEN"
  L=$(cat "$OUT/dirs/e3_best_layer.txt")
  conditions "$OUT/conditions.json" "[
    {'name': 'control', 'direction': None, 'alpha': 0.0, 'layer': 40, 'module': 'down_proj'},
    {'name': 'refl_s0.5', 'projector': {'direction': '$OUT/dirs/refl_L$L.npz', 'layer': $L, 'strength': 0.5}, 'direction': None, 'alpha': 0.0, 'layer': 40, 'module': 'down_proj'},
    {'name': 'refl_s1.0', 'projector': {'direction': '$OUT/dirs/refl_L$L.npz', 'layer': $L, 'strength': 1.0}, 'direction': None, 'alpha': 0.0, 'layer': 40, 'module': 'down_proj'},
    {'name': 'rand_s1.0', 'projector': {'direction': '$OUT/dirs/refl_rand_L$L.npz', 'layer': $L, 'strength': 1.0}, 'direction': None, 'alpha': 0.0, 'layer': 40, 'module': 'down_proj'}]"
  generate "$OUT/conditions.json" "$OUT/gen.jsonl" "$DATA/prompts.jsonl" "$EVAL_DOCS" "$MAX_TOKENS" live
  ;;
E4)
  if [ -f "$OUT/result_A.json" ]; then
    echo "== variant A already fitted: $OUT/result_A.json"
  else
    gpu $PY probe.py --stage features --variant A --eval-prompts "$DATA/prompts.jsonl" --out-dir "$OUT" --max-model-len "$MAX_MODEL_LEN"
    $PY probe.py --stage fit --variant A --out-dir "$OUT"
  fi
  until [ -f "$WORK/out/T0/DONE" ]; do echo "== waiting for T0 traces ($(date +%T))"; sleep 300; done
  gpu $PY probe.py --stage features --variant B --eval-prompts "$DATA/prompts.jsonl" --out-dir "$OUT" \
    --train-traces "$WORK/out/T0/train_traces.jsonl" --eval-traces "$WORK/out/T0/eval_traces.jsonl" --max-model-len "$MAX_MODEL_LEN"
  $PY probe.py --stage fit --variant B --out-dir "$OUT"
  ;;
E5)
  until [ -f "$WORK/out/T0/DONE" ]; do echo "== waiting for T0 traces ($(date +%T))"; sleep 300; done
  gpu $PY reflection.py --stage e5 --prompts "$WORK/out/T0/train_prompts.jsonl" --traces "$WORK/out/T0/train_traces.jsonl" \
    --out-dir "$OUT/dirs" --max-model-len "$MAX_MODEL_LEN"
  conditions "$OUT/conditions.json" "[
    {'name': 'control', 'direction': None, 'alpha': 0.0, 'layer': 40, 'module': 'down_proj'},
    {'name': 'heads_edit', 'heads': {'edit': '$OUT/dirs/heads_edit.npz', 'strength': 1.0}, 'direction': None, 'alpha': 0.0, 'layer': 40, 'module': 'down_proj'},
    {'name': 'heads_random', 'heads': {'edit': '$OUT/dirs/heads_edit_random.npz', 'strength': 1.0}, 'direction': None, 'alpha': 0.0, 'layer': 40, 'module': 'down_proj'}]"
  generate "$OUT/conditions.json" "$OUT/gen.jsonl" "$DATA/prompts.jsonl" "$EVAL_DOCS" "$MAX_TOKENS" live
  ;;
*) echo "unknown experiment $EXP" >&2; exit 2 ;;
esac
echo "== $EXP done $(date -Is)"
