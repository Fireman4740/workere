"""Read-only technical parity with the exact archived T2 operator.

The archived script is parsed, never imported or executed: importing it would
run its gold-based scoring. Its function must be AST-equal to the reviewed pure
numerical reference below. This validation reads no relation labels.
"""

from __future__ import annotations

import ast
import inspect
from collections import defaultdict
from pathlib import Path

from .historical import native_t2
from .schema import require, text_hash


def propagate(keys, score, groups_of):
    """Max over every mention pair linking the same two predicted groups, then broadcast."""
    member = {}
    for doc, groups in groups_of.items():
        m = {x: f"g{i}" for i, g in enumerate(groups) for x in g}
        member[doc] = m
    buckets = defaultdict(list)
    for i, (doc, a, b) in enumerate(keys):
        ga, gb = (
            member.get(doc, {}).get(a, "s" + a),
            member.get(doc, {}).get(b, "s" + b),
        )
        if ga != gb:
            buckets[(doc, frozenset((ga, gb)))].append(i)
    out = score.copy()
    for idx in buckets.values():
        out[idx] = score[idx].max()
    return out


REFERENCE = inspect.getsource(propagate)


def historical_t2_parity(bundle: dict, groups: dict, archived_script: Path) -> dict:
    import numpy as np

    source = archived_script.read_text()
    found = [
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == "propagate"
    ]
    expected = ast.parse(REFERENCE).body[0]
    require(
        len(found) == 1 and ast.dump(found[0]) == ast.dump(expected),
        "HISTORICAL_OPERATOR_DRIFT",
    )
    current = native_t2(bundle, groups)
    checked = []
    for did, rows in bundle["scores"].items():
        if groups[did]["status"] != "AVAILABLE":
            continue
        keys, values = [], []
        for row in rows:
            a, b = row["pair"]
            keys.extend([(did, a, b), (did, b, a)])
            values.extend([row["fusion"], row["fusion"]])
        scores = propagate(keys, np.array(values), {did: groups[did]["groups"]})
        reference_positive = {
            tuple(sorted((a, b))) for (_, a, b), s in zip(keys, scores) if s >= 0.39
        }
        ours = {tuple(p) for p in current["documents"][did]["positive_pairs"]}
        require(reference_positive == ours, "T2_NATIVE_PARITY_FAILED:" + did)
        checked.append(did)
    return {
        "schema": "eci-proof-technical-parity/v1",
        "exact_T2_matches": len(checked),
        "checked_documents": checked,
        "missing_documents": [
            d for d, g in groups.items() if g["status"] != "AVAILABLE"
        ],
        "reviewed_reference_ast_equal": True,
        "historical_script_sha256_utf8": text_hash(source),
        "gold_read": False,
        "model_calls": 0,
    }
