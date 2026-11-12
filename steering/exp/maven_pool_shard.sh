#!/usr/bin/env bash
# One CC-IN2P3 L40S: start HyperQwen in throughput (batch) mode on the W4A16 checkpoint, wait
# until it serves, run this shard's MAVEN pool reads at high concurrency, then stop the server.
# Runs inside an srun allocation. Usage: maven_pool_shard.sh <shard> <shards>
set -uo pipefail
SHARD=${1:?shard} SHARDS=${2:?shards}
ROOT=${ROOT:-/sps/liris/mcarlesso/maven-pool}
REPO=$ROOT/repo OUT=$ROOT/out LOG=$ROOT/logs/shard-$SHARD
SIF=/pbs/throng/liris/mcarlesso/steer/hyperqwen.sif
W4=/pbs/throng/liris/mcarlesso/steer/models/Qwen3.8-27B-W4A16-AutoRound
PORT=$((8100 + SHARD))
CONC=${CONC:-64}
mkdir -p "$LOG"
# Compile caches go to node-local scratch: on the shared NFS home, concurrent shards hit
# "Stale file handle" and the 20 GB home quota filled ("No space left on device").
CACHE=${TMPDIR:-/tmp}/mp-cache-$SHARD && mkdir -p "$CACHE"
CACHE_ENV="XDG_CACHE_HOME=$CACHE,VLLM_CACHE_ROOT=$CACHE/vllm,TRITON_CACHE_DIR=$CACHE/triton"
CACHE_ENV="$CACHE_ENV,TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor,FLASHINFER_WORKSPACE_BASE=$CACHE"
BIND=(-B "$W4:/app/models/Qwen3.8-27B-W4A16-AutoRound:ro" -B "$ROOT" -B "$CACHE")

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > "$LOG/gpu.txt"
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits -l 5 \
  > "$LOG/gpu.csv" 2>/dev/null & SMI=$!
setsid apptainer exec --nv "${BIND[@]}" \
  --env MAX_SEQS=$CONC,MAX_LEN=32768,PORT=$PORT,BIND_HOST=127.0.0.1,HOST=127.0.0.1,PREPARE=0,VERIFY=0 \
  --env VLLM_USE_FLASHINFER_SAMPLER=0,FLASHINFER_DISABLE_VERSION_CHECK=1,$CACHE_ENV \
  "$SIF" bash -c 'cd /app && exec bash batch/start_qwen.sh' > "$LOG/server.log" 2>&1 & SRV=$!
cleanup() { kill -- -"$SRV" 2>/dev/null; sleep 5; kill -9 -- -"$SRV" 2>/dev/null; kill "$SMI" 2>/dev/null; }
trap cleanup EXIT

t0=$(date +%s)
until curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null; do
  kill -0 "$SRV" 2>/dev/null || { echo "server died during boot"; tail -30 "$LOG/server.log"; exit 3; }
  [ $(( $(date +%s) - t0 )) -gt 3600 ] && { echo "server not ready after 3600 s"; exit 4; }
  sleep 10
done
echo "server ready after $(( $(date +%s) - t0 )) s on $(hostname)"

apptainer exec --nv "${BIND[@]}" --env PYTHONPATH="$REPO",HF_HOME="$ROOT/hf-home",$CACHE_ENV "$SIF" \
  bash -c "cd $REPO && /app/venv/bin/python -u -m steering.maven_pool run --out $OUT \
    --url http://127.0.0.1:$PORT/v1 --shard $SHARD --shards $SHARDS --concurrency $CONC"
echo "client rc=$? $(date -Is)"
