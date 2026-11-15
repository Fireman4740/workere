#!/usr/bin/env bash
# Runs inside the pinned HyperQwen batch container; one shared RTX3090.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
PY=/app/venv/bin/python
export PYTHONPATH=$REPO
: "${ORX_RUN_ID:?tracked execution required}" "${ORX_COMMIT:?snapshot required}"
eval "$("$PY" - <<'PY'
import json,shlex
c=json.load(open('steering/maven_replay_config.json'))
for k,v in {'ARM':c['arm'],'ROOT':c['campaign_root'],'MODEL':c['model_path'],
            'MAX_LEN':c['engine']['max_model_len'],'MAX_SEQS':c['engine']['concurrency'],
            'GPU_UTIL':c['engine']['gpu_memory_utilization'],'MAX_LEASES':c['max_gpu_leases'],
            'INT8_ACT':c['engine']['int8_act'],'INT8_LAYERS':c['engine']['int8_layers']}.items():
 print(k+'='+shlex.quote(str(v)))
PY
)"
OUT=$ROOT/$ARM
ATTEMPT=$OUT/logs/$ORX_RUN_ID
mkdir -p "$ATTEMPT" /cache/maven-replay-3090
echo "MC_CONTROLLER_3090 arm=$ARM run=$ORX_RUN_ID commit=$ORX_COMMIT"
exec 9>"$OUT/allocation.lock"
flock -n 9 || { echo 'another controller owns this experiment'; exit 2; }
"$PY" -m unittest steering.test_maven_replay steering.test_maven_budget steering.test_maven_pool
"$PY" -u -m steering.maven_replay wait-source
"$PY" -u -m steering.maven_replay prepare
exec 8>/cache/maven-replay-3090/gpu.lock
flock 8
"$PY" - <<'PY'
import torch,json
assert torch.cuda.device_count()==1
name=torch.cuda.get_device_name(0)
assert '3090' in name,name
print('MC_GPU',json.dumps(dict(name=name,count=1)),flush=True)
PY
export MB_API_KEY=$("$PY" -c 'import secrets; print(secrets.token_hex(32))')
export VLLM_API_KEY=$MB_API_KEY
export MAX_LEN MAX_SEQS GPU_UTIL MODEL INT8_ACT INT8_LAYERS
export PORT=18120 HOST=127.0.0.1 BIND_HOST=127.0.0.1 PREPARE=0 VERIFY=0
export KV=fp8 VISION=0 TOOLS=0 VLLM_OFFLOAD_KEEP_SHM=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export VLLM_USE_FLASHINFER_SAMPLER=0 FLASHINFER_DISABLE_VERSION_CHECK=1
export XDG_CACHE_HOME=/cache/maven-replay-3090
export VLLM_CACHE_ROOT=$XDG_CACHE_HOME/vllm
export TRITON_CACHE_DIR=$XDG_CACHE_HOME/triton
export TORCHINDUCTOR_CACHE_DIR=$XDG_CACHE_HOME/inductor
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
nvidia-smi --query-gpu=timestamp,uuid,memory.used,utilization.gpu --format=csv,noheader,nounits \
  -l 10 > "$ATTEMPT/gpu.csv" 2>/dev/null & SMI=$!
setsid bash /app/batch/start_qwen.sh > "$ATTEMPT/server.log" 2>&1 & SRV=$!
start=$(date +%s)
until curl -sf -H "Authorization: Bearer $MB_API_KEY" "http://127.0.0.1:$PORT/v1/models" > "$ATTEMPT/models.json"; do
  kill -0 "$SRV" 2>/dev/null || { tail -60 "$ATTEMPT/server.log"; exit 4; }
  [ $(( $(date +%s)-start )) -lt 3600 ] || { echo 'batch startup timeout'; exit 5; }
  sleep 10
done
echo "MC_SERVER_READY arm=$ARM RTX3090 batch max_len=$MAX_LEN concurrency=$MAX_SEQS"
for lease in $(seq 1 "$MAX_LEASES"); do
  export REPLAY_LEASE_ID=ssh-$lease
  set +e
  "$PY" -u -m steering.maven_replay run --url "http://127.0.0.1:$PORT/v1"
  RC=$?
  set -e
  if [ "$RC" -eq 75 ]; then echo "MC_CHECKPOINT lease=$lease responses retained"; continue; fi
  echo "MC_EXIT arm=$ARM rc=$RC"
  exit "$RC"
done
echo 'MC_INCOMPLETE planned duration ceiling reached; no partial score'
exit 76
