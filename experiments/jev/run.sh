#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
if [[ -f n1-anchoring/frozen/protocol.json ]]; then
  exec uv run --isolated --no-project --python 3.13 python n1-anchoring/run.py
fi
if [[ -f m1-diversity/frozen/m1-protocol-frozen.json ]]; then
  exec uv run --isolated --no-project --python 3.13 python m1-diversity/run.py
fi
# The launcher forwards credentials from its environment; no key is written in source.
# M0 retains this inherited entry point; protocol/stage are versioned, never env overrides.
if [[ -f eci_m0_pilot/config.json ]]; then
  uv run --isolated --no-project --python 3.13 python -m unittest discover -s eci_m0_pilot -p 'test_*.py' -v
  exec uv run --isolated --no-project --python 3.13 python eci_m0_pilot/run_pilot.py
fi
uv run --project benchmark --locked --extra hub python -m pipelines.jev.cli dry-run \
  --config experiments/jev/active.yaml --output logs/jev-plan.json
uv run --project benchmark --locked --extra hub python -m pipelines.jev.cli run \
  --config experiments/jev/active.yaml --output logs/jev-receipt.json
