#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
# Optional owner-only credential provisioned on the user's execution host.
if [[ -z "${OPENROUTER_API_KEY:-}${OPENROUTER:-}" && -f "$HOME/.config/workere/openrouter.key" ]]; then
  export OPENROUTER_API_KEY="$(cat "$HOME/.config/workere/openrouter.key")"
fi
uv run --project benchmark --locked --extra hub python -m pipelines.jev.cli dry-run \
  --config experiments/jev/active.yaml --output logs/jev-plan.json
uv run --project benchmark --locked --extra hub python -m pipelines.jev.cli run \
  --config experiments/jev/active.yaml --output logs/jev-receipt.json
