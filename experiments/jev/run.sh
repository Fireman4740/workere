#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
exec uv run --project benchmark --locked --extra hub python experiments/jev/n3/recover_audit.py
