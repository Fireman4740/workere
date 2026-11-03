#!/usr/bin/env python
"""Alpha sweep at one layer: control first, then each alpha for hint and demo, interleaved."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_sweep(direction_dir: str, layer: int, alphas: list[float], families: list[str]) -> list[dict]:
    conditions = [{"name": "control", "direction": None, "alpha": 0.0, "layer": layer,
                   "module": "down_proj", "positions": "last"}]
    for alpha in alphas:
        for family in families:
            conditions.append({
                "name": f"{family}_L{layer}_a{alpha:g}",
                "direction": str(Path(direction_dir) / f"{family}_L{layer}_down_proj.npz"),
                "alpha": alpha,
                "layer": layer,
                "module": "down_proj",
                "positions": "last",
            })
    return conditions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction-dir", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--alphas", default="0.1,0.05,0.2")
    parser.add_argument("--families", default="hint,demo")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    alphas = [float(a) for a in args.alphas.split(",")]
    conditions = build_sweep(args.direction_dir, args.layer, alphas, args.families.split(","))
    Path(args.out).write_text(json.dumps(conditions, indent=1) + "\n")
    print("conditions:", ", ".join(c["name"] for c in conditions), flush=True)


if __name__ == "__main__":
    main()
