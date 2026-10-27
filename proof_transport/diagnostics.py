"""Predeclared label-free witness-dependence checks, outside the four arms.

These are raw-judge behaviour tests, not benchmark examples or extra additions.
The causal witness is removed while document/mentions remain available. A judge
must not claim to reuse a supplied E that is empty. Structural rejection alone
would be tautological, so we inspect the model's raw causal_reuse obligation.
"""

from __future__ import annotations

from copy import deepcopy

from .core import validate_registry
from .pipeline import collect_items, load_plan, request
from .schema import digest, require


def diagnostic_requests(plan, registry, maximum=20):
    require(type(maximum) is int and 0 < maximum <= 20, "DIAGNOSTIC_LIMIT")
    manifest, bundle, roots, _, config = load_plan(plan)
    validate_registry(bundle, roots, registry)
    selected = sorted(registry["proposals"], key=lambda p: digest(["mask_E", p["id"]]))[
        :maximum
    ]
    requests = []
    for p in selected:
        cert = deepcopy(registry["sources"][p["root_id"]]["certificate"])
        cert["evidence"] = []
        item = {
            "candidate_id": "E_MASK_" + p["id"],
            "cause": p["cause"],
            "alternative": p["alternative"],
            "effect": p["effect"],
            "source_certificate_hash": digest(cert),
            "evidence_hash": digest([]),
            "certificate": cert,
        }
        r = request(
            "A2",
            p["doc"],
            bundle["documents"][p["doc"]],
            [item],
            manifest,
            config,
            registry_hash=digest(registry),
        )
        del r["id"]
        r["stage"] = "diagnostic_E_mask"
        r["diagnostic"] = {
            "original_candidate_id": p["id"],
            "kind": "causal_witness_removed_not_a_new_candidate",
        }
        r["id"] = digest(r)
        requests.append(r)
    return requests


def inspect_diagnostics(plan, registry, receipts):
    requests = diagnostic_requests(plan, registry)
    raw, usage = collect_items(requests, receipts, "candidate_id")
    supported = [
        cid
        for cid, r in raw.items()
        if r.get("obligations", {}).get("causal_reuse") == "SUPPORTED"
    ]
    unknown = [
        cid
        for cid, r in raw.items()
        if r.get("obligations", {}).get("causal_reuse")
        not in {"SUPPORTED", "CONFLICT", "UNKNOWN"}
    ]
    return {
        "schema": "eci-proof-witness-diagnostic/v1",
        "tested": len(raw),
        "claims_reuse_without_witness": supported,
        "unscorable_raw_obligation": unknown,
        "truth_guarantee": False,
        "adds_predictions": False,
        "uses_gold": False,
    }, usage
