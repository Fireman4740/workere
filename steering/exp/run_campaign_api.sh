#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
export PYTHONPATH="$REPO:$REPO/steering"
PY=${PY:-/app/venv/bin/python}
WORK=${WORK:-/app/local_extensions/steering/mech}
OUT="$WORK/out/C3_API/${ORX_RUN_ID:?}"
SOURCE="$WORK/out/C2/${C2_RUN:?}"
NATIVE="$WORK/out/C3/${NATIVE_RUN:?}"
mkdir -p "$OUT"
cd "$REPO"
echo "== C3 API commit=${ORX_COMMIT:-unknown} output=$OUT $(date -Is)"
# This controller uses CPU only. A single persistent batch server owns the GPU.
$PY -u - <<'PY'
import os,time,urllib.request
url=os.environ['API_BASE_URL'].removesuffix('/v1')+'/health'
deadline=time.monotonic()+900
while True:
    try:
        with urllib.request.urlopen(url,timeout=5) as response:
            assert response.status==200
        print('API_READY',url,flush=True)
        break
    except Exception as exc:
        if time.monotonic()>deadline:
            raise RuntimeError('Batch API not healthy after 15 minutes') from exc
        print('WAIT batch API loading',type(exc).__name__,flush=True)
        time.sleep(15)
PY
$PY -u steering/campaign_adjudicate.py prepare --out "$OUT" --source "$SOURCE" --work "$WORK"
$PY -u steering/campaign_api.py --out "$OUT" --native "$NATIVE" --server "$SERVER_METADATA"
echo "== C3 API complete $(date -Is)"
