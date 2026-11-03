#!/usr/bin/env python
"""Paired-document bootstrap of the micro-F1 change of each condition against a control.

Reads the per-document CSV written by score.py. Resampling is over documents, the same
draw for the condition and the control, so the interval is on the paired difference.
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict


def micro_f1(rows) -> float:
    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    return 100 * 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def paired_delta(condition, control, draws=10000, seed=2026110301, level=95.0):
    """(delta, lo, hi) of micro-F1 condition - control over the shared documents."""
    shared = sorted(set(condition) & set(control))
    if not shared:
        raise ValueError("no shared documents")
    delta = micro_f1([condition[d] for d in shared]) - micro_f1([control[d] for d in shared])
    rng = random.Random(seed)
    samples = []
    for _ in range(draws):
        pick = rng.choices(shared, k=len(shared))
        samples.append(micro_f1([condition[d] for d in pick]) - micro_f1([control[d] for d in pick]))
    samples.sort()
    tail = (100 - level) / 200
    return delta, samples[int(tail * draws)], samples[int((1 - tail) * draws) - 1]


def load(path, subset):
    by_condition = defaultdict(dict)
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            if row["subset"] != subset:
                continue
            by_condition[row["condition"]][row["doc"]] = {
                k: int(row[k]) for k in ("tp", "fp", "fn")
            }
    return by_condition


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--control", default="control")
    parser.add_argument("--subset", default="qwen_47")
    parser.add_argument("--draws", type=int, default=10000)
    args = parser.parse_args()

    by_condition = load(args.csv, args.subset)
    control = by_condition.get(args.control)
    if not control:
        raise SystemExit(f"control condition {args.control!r} not in {args.csv}")
    print(f"subset={args.subset} control={args.control} F1={micro_f1(list(control.values())):.2f}")
    for name, rows in sorted(by_condition.items()):
        if name == args.control:
            continue
        delta, lo, hi = paired_delta(rows, control, args.draws)
        flag = "" if lo <= 0 <= hi else "  *"
        print(f"{name:24s} docs={len(set(rows) & set(control)):3d} dF1={delta:+6.2f} [{lo:+6.2f}, {hi:+6.2f}]{flag}")


if __name__ == "__main__":
    main()
