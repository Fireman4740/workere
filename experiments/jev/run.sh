#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
uv run --project benchmark --locked --extra hub python -m pipelines.jev.n9
