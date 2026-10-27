#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
# The service credential is read on the execution host; never copied into source.
export PYTHONUNBUFFERED=1
exec uv run --locked python experiments/jev/qwen/run.py
