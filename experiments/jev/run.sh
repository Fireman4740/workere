#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
# Experimental child only: supervised authorized proof-transport sidecar.
# Parent baseline and its active.yaml stay immutable; never dispatch them here.
# Same locked benchmark+hub environment and fixed command as the parent.
exec uv run --project benchmark --locked --extra hub python \
  -m experiments.jev.proof_transport_launch.controller
