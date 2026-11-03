#!/usr/bin/env python
"""Write control directions next to a real one (numpy only, no GPU).

  <name>_neg.npz       the same direction with the sign flipped
  <name>_rand<k>.npz   random unit directions, same dimension and same ref_norm

A real effect must beat the random directions and must reverse with the sign flip.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def random_unit(dim: int, seed: int) -> np.ndarray:
    vector = np.random.default_rng(seed).standard_normal(dim)
    return (vector / np.linalg.norm(vector)).astype(np.float32)


def write_direction(path: Path, vector: np.ndarray, source) -> None:
    np.savez(
        path,
        v=vector,
        ref_norm=source["ref_norm"],
        layer=source["layer"],
        module=source["module"],
        method=np.array(path.stem),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", required=True)
    parser.add_argument("--random", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261103)
    args = parser.parse_args()

    source = np.load(args.direction)
    base = Path(args.direction)
    vector = source["v"]
    write_direction(base.with_name(base.stem + "_neg.npz"), -vector, source)
    for k in range(args.random):
        write_direction(
            base.with_name(f"{base.stem}_rand{k}.npz"),
            random_unit(vector.shape[0], args.seed + k),
            source,
        )
    print(f"controls written next to {base}: neg + {args.random} random")


if __name__ == "__main__":
    main()
