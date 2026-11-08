#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
export PYTHONPATH="$REPO:$REPO/steering"
PY=${PY:-/app/venv/bin/python}
cd "$REPO"
exec "$PY" -u -m steering.ap_campaign --stage "${AP_STAGE:?}"
