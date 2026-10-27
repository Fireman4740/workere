"""Exact native T2 group-pair max replay, without labels or new grouping.

This diagnostic is NOT the controlled A3 arm: it uses all original scores and
both historical groups, has no source-selection/cap projection, and may expose
many more pairs. The old code leaves in-group scores unchanged (neither
suppressing nor max-propagating them).
"""

from __future__ import annotations

from collections import defaultdict


def native_t2(bundle: dict, groups: dict) -> dict:
    out = {}
    for did, rows in bundle["scores"].items():
        cached = groups[did]
        if cached["status"] != "AVAILABLE":
            out[did] = dict(cached)
            continue
        membership = {m: f"g{i}" for i, g in enumerate(cached["groups"]) for m in g}

        def member(m, membership=membership):
            return membership.get(m, "singleton:" + m)

        best = defaultdict(float)
        for row in rows:
            a, b = row["pair"]
            if member(a) != member(b):
                key = tuple(sorted((member(a), member(b))))
                best[key] = max(best[key], row["fusion"])
        predictions = []
        for row in rows:
            a, b = row["pair"]
            score = (
                row["fusion"]
                if member(a) == member(b)
                else best[tuple(sorted((member(a), member(b))))]
            )
            if score >= 0.39:
                predictions.append(row["pair"])
        out[did] = {
            "status": "AVAILABLE",
            "positive_pairs": predictions,
            "operator": "historical_group_pair_max_no_in_group_suppression",
        }
    return {
        "schema": "eci-proof-T2-native-diagnostic/v1",
        "documents": out,
        "is_controlled_A3": False,
        "model_calls": 0,
        "training": False,
        "threshold": 0.39,
    }
