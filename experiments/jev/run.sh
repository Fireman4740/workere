#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
# The launcher forwards credentials from its environment; no key is written in source.
uv run --project benchmark --locked --extra hub python -m pipelines.jev.cli dry-run \
  --config experiments/jev/active.yaml --output logs/jev-plan.json
uv run --project benchmark --locked --extra hub python -m pipelines.jev.cli run \
  --config experiments/jev/active.yaml --output logs/jev-receipt.json
