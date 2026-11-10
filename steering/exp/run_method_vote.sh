#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
export PYTHONPATH="$REPO:$REPO/steering"
cd "$REPO"
exec "${PY:-/app/venv/bin/python}" -u -m steering.method_vote
