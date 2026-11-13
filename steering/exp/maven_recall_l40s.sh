#!/usr/bin/env bash
# Fixed entrypoint runs in a tracked htc_daemon CPU job; only its srun gets one L40S.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
PY_HOST=$(command -v python3)
eval "$("$PY_HOST" - <<'PY'
import json,shlex
c=json.load(open('steering/maven_recall_config.json'))
for k,v in {'ARM':c['arm'], 'ROOT':c['campaign_root'], 'SIF':c['container'],
            'W4':c['model_path'], 'CONC':c['engine']['concurrency'],
            'MAX_LEN':c['engine']['max_model_len'], 'INT8_ACT':c['engine']['int8_act'],
            'INT8_LAYERS':c['engine']['int8_layers']}.items():
 print(k+'='+shlex.quote(str(v)))
PY
)"
: "${ORX_RUN_ID:?tracked run required}" "${ORX_COMMIT:?committed snapshot required}"
# Use a private ephemeral key shared by this server and client; override the
# container's unrelated api_key.txt without ever printing either key.
export MB_API_KEY=${MB_API_KEY:-$("$PY_HOST" -c 'import secrets; print(secrets.token_hex(32))')}
OUT=$ROOT/$ARM
ATTEMPT=$OUT/logs/$ORX_RUN_ID
mkdir -p "$ATTEMPT" "$ROOT/hf-home" "$ROOT/apptainer-cache" "$ROOT/cache"
export APPTAINER_CACHEDIR=$ROOT/apptainer-cache
BIND=(-B "$REPO:$REPO:ro" -B "$ROOT"
      -B /sps/liris/mcarlesso/maven-pool/out:/sps/liris/mcarlesso/maven-pool/out:ro
      -B "$W4:$W4:ro" -B "$W4:/app/models/Qwen3.8-27B-W4A16-AutoRound:ro"
      -B "$SIF:$SIF:ro")

if [ "${1:-}" != _gpu ]; then
  echo "MR_CONTROLLER arm=$ARM run=$ORX_RUN_ID commit=$ORX_COMMIT host=$(hostname)"
  # A resumable campaign keeps exclusive ownership of an arm, including the GPU allocation.
  exec 9>"$OUT/allocation.lock"
  flock -n 9 || { echo 'another job owns this arm'; exit 2; }
  apptainer exec "${BIND[@]}" --env PYTHONPATH="$REPO" "$SIF" \
    /app/venv/bin/python -m unittest steering.test_maven_budget steering.test_maven_pool steering.test_maven_recall
  apptainer exec "${BIND[@]}" --env PYTHONPATH="$REPO",HF_HOME="$ROOT/hf-home" "$SIF" \
    /app/venv/bin/python -u -m steering.maven_recall prepare
  # Do not accidentally make a GPU step inside the CPU allocation.
  for v in $(compgen -v SLURM_); do unset "$v"; done
  CHILD=''
  stop_child() {
    trap - TERM INT
    if [ -n "$CHILD" ]; then kill -TERM "$CHILD" 2>/dev/null || true; wait "$CHILD" || true; fi
    if [ -f "$ATTEMPT/gpu_job_id" ]; then scancel "$(cat "$ATTEMPT/gpu_job_id")" 2>/dev/null || true; fi
    exit 143
  }
  trap stop_child TERM INT
  srun --partition=gpu_l40s_interactive --qos=gpu --account=liris --gpus=1 --ntasks=1 \
    --cpus-per-task=8 --mem=96G --time=1-00:00:00 --licenses=sps --job-name="mr-$ARM" \
    bash "$REPO/steering/exp/maven_recall_l40s.sh" _gpu &
  CHILD=$!
  set +e
  wait "$CHILD"
  RC=$?
  set -e
  trap - TERM INT
  if [ -f "$ATTEMPT/gpu_job_id" ]; then
    sacct -j "$(cat "$ATTEMPT/gpu_job_id")" --parsable2 \
      -o JobID,State,ElapsedRaw,AllocTRES,ExitCode > "$ATTEMPT/accounting.txt" || true
  fi
  echo "MR_EXIT arm=$ARM rc=$RC"
  exit "$RC"
fi

printf '%s\n' "$SLURM_JOB_ID" > "$ATTEMPT/gpu_job_id"
scontrol show job "$SLURM_JOB_ID" > "$ATTEMPT/allocation.txt"
CACHE=$(mktemp -d "$ROOT/cache/mr-$ORX_RUN_ID.XXXXXX")
BIND+=(-B "$CACHE")
CACHE_ENV="XDG_CACHE_HOME=$CACHE,VLLM_CACHE_ROOT=$CACHE/vllm,TRITON_CACHE_DIR=$CACHE/triton"
CACHE_ENV="$CACHE_ENV,TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor,FLASHINFER_WORKSPACE_BASE=$CACHE"
# Port leases are local to the node and remain locked for the server's lifetime.
PORT=''
for candidate in $(seq 18100 18299); do
  exec 8>"/tmp/mr-port-$candidate.lock"
  if flock -n 8 && "$PY_HOST" - "$candidate" <<'PY'
import socket,sys
with socket.socket() as s:
 try:s.bind(('127.0.0.1',int(sys.argv[1])))
 except OSError:sys.exit(1)
PY
  then PORT=$candidate; break; fi
  exec 8>&-
done
[ -n "$PORT" ] || { echo 'no free server port'; exit 3; }
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:?missing Slurm GPU binding}
apptainer exec --nv "${BIND[@]}" "$SIF" /app/venv/bin/python - <<'PY'
import torch,json
assert torch.cuda.device_count()==1, 'exactly one visible GPU required'
name=torch.cuda.get_device_name(0)
assert 'L40S' in name, name
print('MR_GPU',json.dumps({'name':name,'count':torch.cuda.device_count()}),flush=True)
PY
SRV='' SMI=''
cleanup() {
  rc=$?
  trap - EXIT TERM INT
  if [ -n "$SRV" ]; then
    kill -- -"$SRV" 2>/dev/null || true
    sleep 3
    kill -9 -- -"$SRV" 2>/dev/null || true
  fi
  if [ -n "$SMI" ]; then kill "$SMI" 2>/dev/null || true; fi
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 143' TERM INT
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=timestamp,uuid,memory.used,utilization.gpu \
  --format=csv,noheader,nounits -l 10 > "$ATTEMPT/gpu.csv" 2>/dev/null & SMI=$!
setsid apptainer exec --nv "${BIND[@]}" \
  --env MAX_SEQS="$CONC",MAX_LEN="$MAX_LEN",PORT="$PORT",HOST=127.0.0.1,BIND_HOST=127.0.0.1,PREPARE=0,VERIFY=0 \
  --env VLLM_API_KEY="$MB_API_KEY" \
  --env KV=fp8,GPU_UTIL=0.94,INT8_ACT="$INT8_ACT",INT8_LAYERS="$INT8_LAYERS",VLLM_OFFLOAD_KEEP_SHM=1 \
  --env VLLM_USE_FLASHINFER_SAMPLER=0,FLASHINFER_DISABLE_VERSION_CHECK=1,"$CACHE_ENV" \
  "$SIF" bash -c 'cd /app && exec bash batch/start_qwen.sh' > "$ATTEMPT/server.log" 2>&1 & SRV=$!
start=$(date +%s)
until curl -sf -H "Authorization: Bearer $MB_API_KEY" "http://127.0.0.1:$PORT/v1/models" > "$ATTEMPT/models.json"; do
  kill -0 "$SRV" 2>/dev/null || { tail -50 "$ATTEMPT/server.log"; exit 4; }
  [ $(( $(date +%s)-start )) -lt 3600 ] || { echo 'server startup timeout'; exit 5; }
  sleep 10
done
echo "MR_SERVER_READY arm=$ARM port=$PORT host=$(hostname) max_len=$MAX_LEN concurrency=$CONC"
apptainer exec --nv "${BIND[@]}" --env PYTHONPATH="$REPO",HF_HOME="$ROOT/hf-home","$CACHE_ENV" \
  "$SIF" /app/venv/bin/python -u -m steering.maven_recall run --url "http://127.0.0.1:$PORT/v1"
