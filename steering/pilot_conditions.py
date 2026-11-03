#!/usr/bin/env python
"""Write the ordered condition list for the N13 CTB50 steering pilot.

Stage 1 is the cheap screen: the unsteered offline control and one real direction per
family, in the agreed order hint -> demo -> self. Stage 2 adds, for each family, the
sign-flipped and random-direction controls; run it only for families whose stage-1 F1
moved away from the control.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

FAMILIES = ("hint", "demo", "self")


def build_conditions(direction_dir, layer, module, alpha, positions="last", n_random=2):
    direction_dir = Path(direction_dir)

    def entry(name, stem, a, stage):
        return {
            "name": name,
            "stage": stage,
            "direction": str(direction_dir / f"{stem}.npz") if stem else None,
            "alpha": a,
            "layer": layer,
            "module": module,
            "positions": positions,
        }

    conditions = [entry("control", None, 0.0, 1)]
    for family in FAMILIES:
        stem = f"{family}_L{layer}_{module}"
        conditions.append(entry(f"{family}_pos", stem, alpha, 1))
    for family in FAMILIES:
        stem = f"{family}_L{layer}_{module}"
        conditions.append(entry(f"{family}_neg", f"{stem}_neg", alpha, 2))
        for k in range(n_random):
            conditions.append(entry(f"{family}_rand{k}", f"{stem}_rand{k}", alpha, 2))
    return conditions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction-dir", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--module", default="down_proj")
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--positions", default="last")
    parser.add_argument("--random", type=int, default=2)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    conditions = build_conditions(
        args.direction_dir, args.layer, args.module, args.alpha, args.positions, args.random
    )
    Path(args.out).write_text(json.dumps(conditions, indent=1) + "\n")
    for c in conditions:
        print(f"stage {c['stage']}  {c['name']}")


if __name__ == "__main__":
    main()
