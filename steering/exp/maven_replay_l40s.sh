#!/usr/bin/env bash
# Tracked CPU controller; each srun gets one preflighted L40S or H100.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
PY_HOST=$(command -v python3)
eval "$("$PY_HOST" - <<'PY'
import json,shlex
from steering.maven_replay_allocation import gpu_resources
c=json.load(open('steering/maven_replay_config.json'))
for k,v in {'ARM':c['arm'], 'ROOT':c['campaign_root'], 'SIF':c['container'],
            'W4':c['model_path'], 'CONC':c['engine']['concurrency'],
            'MAX_LEN':c['engine']['max_model_len'], 'INT8_ACT':c['engine']['int8_act'],
            'INT8_LAYERS':c['engine']['int8_layers'], 'MAX_LEASES':c['max_gpu_leases'],
            'ALLOCATION_SECONDS':c.get('allocation_seconds',86400),
            'ALLOCATION_TIME':f"{c.get('allocation_seconds',86400)//3600:02d}:"
              f"{c.get('allocation_seconds',86400)%3600//60:02d}:"
              f"{c.get('allocation_seconds',86400)%60:02d}", **gpu_resources(c)}.items():
 print(k+'='+shlex.quote(str(v)))
PY
)"
: "${ORX_RUN_ID:?tracked run required}" "${ORX_COMMIT:?committed snapshot required}"
# Use a private ephemeral key shared by this server and client; override the
# container's unrelated api_key.txt without ever printing either key.
export MB_API_KEY=${MB_API_KEY:-$("$PY_HOST" -c 'import secrets; print(secrets.token_hex(32))')}
OUT=$ROOT/$ARM
ATTEMPT=$OUT/logs/$ORX_RUN_ID
mkdir -p "$ATTEMPT" "$ROOT/hf-home" "$ROOT/apptainer-cache"
export APPTAINER_CACHEDIR=$ROOT/apptainer-cache
BIND=(-B "$REPO:$REPO:ro" -B "$ROOT"
      -B /sps/liris/mcarlesso/maven-budget-20261113/D:/sps/liris/mcarlesso/maven-budget-20261113/D:ro
      -B /sps/liris/mcarlesso/maven-pool/out:/sps/liris/mcarlesso/maven-pool/out:ro
      -B "$W4:$W4:ro" -B "$W4:/app/models/Qwen3.8-27B-W4A16-AutoRound:ro"
      -B "$SIF:$SIF:ro")

if [ "${1:-}" != _gpu ]; then
  echo "MC_CONTROLLER arm=$ARM run=$ORX_RUN_ID commit=$ORX_COMMIT host=$(hostname)"
  # A resumable campaign keeps exclusive ownership of an arm, including the GPU allocation.
  exec 9>"$OUT/allocation.lock"
  flock -n 9 || { echo 'another job owns this arm'; exit 2; }
  apptainer exec "${BIND[@]}" --env PYTHONPATH="$REPO" "$SIF" \
    /app/venv/bin/python -m unittest steering.test_maven_replay steering.test_maven_budget steering.test_maven_pool steering.test_maven_max_context
  apptainer exec "${BIND[@]}" --env PYTHONPATH="$REPO" "$SIF" \
    /app/venv/bin/python -u -m steering.maven_replay wait-source
  apptainer exec "${BIND[@]}" --env PYTHONPATH="$REPO",HF_HOME="$ROOT/hf-home" "$SIF" \
    /app/venv/bin/python -u -m steering.maven_replay prepare
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
  for lease in $(seq 1 "$MAX_LEASES"); do
    echo "MC_LEASE arm=$ARM lease=$lease maximum=$MAX_LEASES"
    rm -f "$ATTEMPT/gpu_job_id"
    srun --partition="$GPU_PARTITION" --qos=gpu --account=liris --gpus=1 --ntasks=1 \
      --cpus-per-task="$GPU_CPUS" --mem="$GPU_MEM" --time="$ALLOCATION_TIME" --licenses=sps --job-name="mc-$ARM" \
      bash "$REPO/steering/exp/maven_replay_l40s.sh" _gpu &
    CHILD=$!
    set +e
    wait "$CHILD"
    RC=$?
    set -e
    CHILD=''
    gpu_job=''
    ACCOUNTING=$ATTEMPT/accounting-lease-$lease.txt
    : > "$ACCOUNTING"
    STATE=''
    if [ -f "$ATTEMPT/gpu_job_id" ]; then
      gpu_job=$(cat "$ATTEMPT/gpu_job_id")
      # Accounting may lag process exit; never confuse a timed-out step with
      # an OOM or an explicit cancellation of its parent allocation.
      for poll in 1 2 3 4 5; do
        sacct -j "$gpu_job" --parsable2 -o JobID,State,ElapsedRaw,AllocTRES,ExitCode \
          > "$ACCOUNTING" || true
        STATE=$("$PY_HOST" "$REPO/steering/maven_replay_allocation.py" \
          --accounting "$ACCOUNTING" --job-id "$gpu_job" --field state)
        case "$STATE" in ''|PENDING|RUNNING|COMPLETING) sleep 2;; *) break;; esac
      done
      cp "$ACCOUNTING" "$ATTEMPT/accounting-$gpu_job.txt"
    fi
    REASON=$("$PY_HOST" "$REPO/steering/maven_replay_allocation.py" \
      --accounting "$ACCOUNTING" --job-id "$gpu_job" --return-code "$RC" --field reason)
    echo "MC_ALLOCATION arm=$ARM lease=$lease job=$gpu_job rc=$RC scheduler_state=$STATE"
    if [ -n "$REASON" ]; then
      echo "MC_CHECKPOINT arm=$ARM lease=$lease reason=$REASON; responses retained, requesting next allocation"
      continue
    fi
    echo "MC_EXIT arm=$ARM rc=$RC"
    exit "$RC"
  done
  echo "MC_INCOMPLETE arm=$ARM planned GPU lease ceiling reached; no final score"
  exit 76
fi

printf '%s\n' "$SLURM_JOB_ID" > "$ATTEMPT/gpu_job_id"
ATTEMPT=$ATTEMPT/$SLURM_JOB_ID
mkdir -p "$ATTEMPT"
scontrol show job "$SLURM_JOB_ID" > "$ATTEMPT/allocation.txt"
export REPLAY_ALLOCATION_END_UNIX=${SLURM_JOB_END_TIME:-$(( $(date +%s)+ALLOCATION_SECONDS ))}
CACHE=$(mktemp -d "${TMPDIR:-/tmp}/mb-$ORX_RUN_ID.XXXXXX")
BIND+=(-B "$CACHE")
CACHE_ENV="XDG_CACHE_HOME=$CACHE,VLLM_CACHE_ROOT=$CACHE/vllm,TRITON_CACHE_DIR=$CACHE/triton"
CACHE_ENV="$CACHE_ENV,TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor,FLASHINFER_WORKSPACE_BASE=$CACHE"
# Port leases are local to the node and remain locked for the server's lifetime.
PORT=''
for candidate in $(seq 18100 18299); do
  exec 8>"/tmp/mb-port-$candidate.lock"
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
apptainer exec --nv "${BIND[@]}" --env EXPECTED_GPU="$GPU_KIND" "$SIF" /app/venv/bin/python - <<'PY'
import torch,json,os
assert torch.cuda.device_count()==1, 'exactly one visible GPU required'
name=torch.cuda.get_device_name(0)
assert os.environ['EXPECTED_GPU'] in name, name
print('MC_GPU',json.dumps({'name':name,'count':torch.cuda.device_count()}),flush=True)
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
  if [ -n "${CACHE:-}" ] && [ -d "$CACHE" ]; then rm -rf -- "$CACHE"; fi
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
echo "MC_SERVER_READY arm=$ARM port=$PORT host=$(hostname) max_len=$MAX_LEN concurrency=$CONC"
apptainer exec --nv "${BIND[@]}" --env PYTHONPATH="$REPO",HF_HOME="$ROOT/hf-home","$CACHE_ENV" \
  --env REPLAY_ALLOCATION_END_UNIX="$REPLAY_ALLOCATION_END_UNIX" \
  "$SIF" /app/venv/bin/python -u -m steering.maven_replay run --url "http://127.0.0.1:$PORT/v1"
