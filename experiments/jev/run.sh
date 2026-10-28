#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
uv run --project benchmark --locked --extra hub python experiments/jev/n8/n8.py
