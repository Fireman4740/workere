#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
# The launcher forwards credentials from its environment; no key is written in source.
if [[ -f experiments/eci_alias_next/workere-run.json ]]; then
  export PYTHONPATH="${PWD}/experiments/eci_alias_next:${PYTHONPATH:-}"
  uv run --project benchmark --locked --extra hub python -m unittest discover \
    -s experiments/eci_alias_next/tests -v
  exec uv run --project benchmark --locked --extra hub python -m eci_alias.workere_run
fi
uv run --project benchmark --locked --extra hub python -m pipelines.jev.cli dry-run \
  --config experiments/jev/active.yaml --output logs/jev-plan.json
uv run --project benchmark --locked --extra hub python -m pipelines.jev.cli run \
  --config experiments/jev/active.yaml --output logs/jev-receipt.json
