"""Deterministic synthetic transport receipts. NOT an ECI model or benchmark."""

from __future__ import annotations

import itertools
from copy import deepcopy
from pathlib import Path

from .pipeline import (
    DEFAULT_CONFIG,
    extraction_requests,
    freeze_registry,
    judge_requests,
    prepare,
    proposal_requests,
    replay,
    write_jsonl_once,
    write_once,
)
from .schema import digest


def citation(doc: dict, quote: str, offset: int = 0) -> dict:
    start = doc["text"].index(quote, offset)
    return {"start": start, "end": start + len(quote), "quote": quote}


def raw_fixture() -> tuple[dict, list]:
    text = "On Monday, the North depot blast caused the evacuation. This explosion occurred at the North depot on Monday."
    tokens = text.split(" ")
    ids = ["a", "b", "ap"]
    spans = [
        [tokens.index("blast")],
        [tokens.index("evacuation.")],
        [tokens.index("explosion")],
    ]
    split = tokens.index("This")
    doc = {
        "id": "SYNTHETIC_SUCCESS",
        "tokens": tokens,
        "mentions": ids,
        "spans": spans,
        "sentences": [[0, split], [split, len(tokens)]],
    }
    controls = [
        {
            "doc": doc["id"],
            "a": a,
            "b": b,
            "luna": 0.9 if {a, b} == {"a", "b"} else 0.0,
            "d1": 0.9 if {a, b} == {"a", "b"} else 0.0,
        }
        for a, b in itertools.permutations(ids, 2)
    ]
    return {"documents": [doc]}, controls


def scope(doc: dict) -> dict:
    return {
        "state": "OBSERVED",
        "holder": "narrator",
        "polarity": "affirmed",
        "world": "actual",
        "modality": "asserted",
        "operators": [],
        "citations": [
            citation(doc, "On Monday, the North depot blast caused the evacuation.")
        ],
    }


def frame(doc: dict) -> dict:
    unknown = {"state": "UNKNOWN", "value": None, "citations": []}
    known = lambda value, quote: {
        "state": "OBSERVED",
        "value": value,
        "citations": [citation(doc, quote)],
    }
    return {
        "participants": unknown,
        "incident": {
            "state": "INFERRED",
            "value": "one_source_occurrence",
            "citations": [citation(doc, "blast")],
        },
        "time": known("Monday", "Monday"),
        "place": known("North depot", "North depot"),
        "granularity": {
            "state": "INFERRED",
            "value": "single_occurrence",
            "citations": [citation(doc, "blast")],
        },
        "scope": scope(doc),
    }


def source_certificate(doc: dict, root: dict) -> dict:
    return {
        "source_id": root["id"],
        "document_sha256": doc["sha256"],
        "cause": "a",
        "effect": "b",
        "evidence": [
            citation(doc, "On Monday, the North depot blast caused the evacuation.")
        ],
        "frames": {"a": frame(doc), "b": frame(doc)},
        "relation_scope": scope(doc),
        "support_mode": "EXPLICIT",
        "justification": "Synthetic explicit causal clause, not a model judgment.",
    }


def a2_response(doc: dict, cert: dict, candidate_id: str) -> dict:
    alternative = deepcopy(frame(doc))
    alternative["incident"]["citations"] = [citation(doc, "This explosion")]
    alternative["granularity"]["citations"] = [citation(doc, "This explosion")]
    return {
        "candidate_id": candidate_id,
        "source_certificate_hash": digest(cert),
        "evidence_hash": digest(cert["evidence"]),
        "binding_kind": "EXPLICIT_ANAPHORA",
        "binding": [citation(doc, "This explosion")],
        "alternative_frame": alternative,
        "relation_scope": scope(doc),
        "discriminating_fields": ["time", "place"],
        "obligations": {
            k: "SUPPORTED"
            for k in (
                "binding",
                "causal_reuse",
                "granularity",
                "scope",
                "discriminants",
            )
        },
        "competitors": {
            "covered_mentions": list(doc["mentions"]),
            "covered_sentences": list(range(len(doc["sentences"]))),
            "residual": "NONE",
            "rivals": [],
        },
        "new_causal_premises": False,
        "justification": "Illustrative anaphoric binding with the exact same source witness.",
    }


def receipt(req: dict, items: list) -> dict:
    return {
        "request_id": req["id"],
        "request_hash": digest(req),
        "model": req["model"],
        "mode": "synthetic_fixture",
        "response": {"items": items},
        "usage": {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
    }


def smoke(output: Path) -> dict:
    suite, controls = raw_fixture()
    cache = output / "synthetic-inputs"
    write_jsonl_once(
        cache / "predictions.jsonl",
        [
            {
                "id": "SYNTHETIC_SUCCESS",
                "metadata": {
                    "coref_groups": [["a", "ap"]],
                    "model": "SYNTHETIC_FIXTURE",
                },
            }
        ],
    )
    config = dict(DEFAULT_CONFIG, max_sources=1)
    prepare(
        suite,
        controls,
        cache,
        output / "plan",
        config=config,
        provenance={"kind": "SYNTHETIC_NO_LABELS_NO_NETWORK"},
    )
    from .pipeline import load_plan

    _, bundle, roots, _, _ = load_plan(output / "plan")
    doc, root = bundle["documents"]["SYNTHETIC_SUCCESS"], roots[0]
    cert = source_certificate(doc, root)
    req0 = extraction_requests(output / "plan")
    r0 = [
        receipt(
            r,
            [
                {
                    "source_id": root["id"],
                    "verdict": "SUPPORTED",
                    "reason": "synthetic fixture",
                    "certificate": cert,
                }
            ],
        )
        for r in req0
    ]
    write_jsonl_once(output / "extract.requests.jsonl", req0)
    write_jsonl_once(output / "extract.receipts.jsonl", r0)
    req1, _ = proposal_requests(output / "plan", r0)
    r1 = [
        receipt(
            r,
            [
                {
                    "source_id": root["id"],
                    "source_certificate_hash": digest(cert),
                    "verdict": "SUPPORTED",
                    "reason": "synthetic fixture",
                    "alternative_ids": ["ap"],
                    "justification": "Synthetic retrieval only.",
                    "citations": cert["evidence"],
                }
            ],
        )
        for r in req1
    ]
    write_jsonl_once(output / "propose.requests.jsonl", req1)
    write_jsonl_once(output / "propose.receipts.jsonl", r1)
    registry = freeze_registry(output / "plan", r0, r1, output / "registry")
    req2 = judge_requests(output / "plan", registry)
    responses = []
    for r in req2:
        items = []
        for p in registry["proposals"]:
            if r["stage"] == "A0":
                item = {
                    "candidate_id": p["id"],
                    "identity": "SAME",
                    "justification": "Synthetic anaphoric identity.",
                }
            elif r["stage"] == "A1":
                item = {
                    "candidate_id": p["id"],
                    "alternative_frame": frame(doc),
                    "discriminating_fields": ["time", "place"],
                    "justification": "Synthetic factual compatibility only.",
                }
            else:
                item = a2_response(doc, cert, p["id"])
            items.append(item)
        responses.append(receipt(r, items))
    write_jsonl_once(output / "judge.requests.jsonl", req2)
    write_jsonl_once(output / "judge.receipts.jsonl", responses)
    run = replay(
        output / "plan", output / "registry", r0, r1, responses, output / "run"
    )
    summary = {
        "kind": "SYNTHETIC_TECHNICAL_SMOKE_NOT_BENCHMARK",
        "labels_read": False,
        "model_calls": 0,
        "tokens": 0,
        "usd": 0,
        "plan_hash": run["plan_hash"],
        "registry_hash": run["registry_hash"],
        "complete": run["complete"],
    }
    write_once(output / "smoke-summary.json", summary)
    return summary
