#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
if [[ -f n6-ctb/frozen/protocol.json ]]; then
  exec uv run --isolated --no-project --python 3.13 python n6-ctb/controller.py
fi
# The launcher forwards credentials from its environment; no key is written in source.
uv run --project benchmark --locked --extra hub python -m pipelines.jev.cli dry-run \
  --config experiments/jev/active.yaml --output logs/jev-plan.json
uv run --project benchmark --locked --extra hub python -m pipelines.jev.cli run \
  --config experiments/jev/active.yaml --output logs/jev-receipt.json
