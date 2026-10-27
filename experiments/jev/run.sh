#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
# Same inherited launch command/environment; this child implements only the user-authorized V0.
# Credentials are forwarded by the launcher, never stored in source.
uv run --project benchmark --locked --extra hub python -m unittest tests.test_agent4_v0 tests.test_agent4_auto
uv run --project benchmark --locked --extra hub python -m unittest discover -s agent4_v0/kit -p tests.py
uv run --project benchmark --locked --extra hub python -m agent4_v0.automated_run
