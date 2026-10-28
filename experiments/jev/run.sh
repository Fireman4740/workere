#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
exec uv run --project benchmark --locked --extra hub python n4-readers/run.py
