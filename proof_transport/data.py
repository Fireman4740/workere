"""One-way allowlist adapters: mixed historical exports -> label-free inference.

This module never imports eci.Doc (which contains gold), changes fusion scores,
normalises IDs numerically, or fills a missing reverse score/group as negative.
"""

from __future__ import annotations

import itertools
import json
from collections import defaultdict, deque
from pathlib import Path

from .schema import (
    digest,
    identifier,
    probability,
    require,
    text_hash,
    validate_document,
)

REVISION = "e6418e8747ac1c7fecf66d56ae41fbde7fd5967a"


def document_from_raw(raw: dict, revision: str = REVISION) -> dict:
    # Deliberately never read raw.text, gold, annots, coref_clusters or mention_type.
    tokens = raw["tokens"]
    require(
        isinstance(tokens, list) and all(isinstance(t, str) and t for t in tokens),
        "TOKENS",
    )
    ids, spans, sentences = raw["mentions"], raw["spans"], raw["sentences"]
    require(
        isinstance(ids, list) and len(ids) == len(spans) and len(ids) == len(set(ids)),
        "MENTION_IDS",
    )
    offsets, pos = [], 0
    for token in tokens:
        offsets.append([pos, pos + len(token)])
        pos += len(token) + 1
    sentence_of = {
        t: i for i, (start, end) in enumerate(sentences) for t in range(start, end)
    }
    mentions = {}
    for mid, indexes in zip(ids, spans):
        identifier(mid)
        require(
            bool(indexes)
            and all(type(t) is int and 0 <= t < len(tokens) for t in indexes),
            "SPAN_INDEX",
        )
        surface = " ".join(tokens[t] for t in indexes)
        if "surfaces" in raw:
            require(raw["surfaces"].get(mid) == surface, "HISTORICAL_SURFACE_MISMATCH")
        if "first_token" in raw:
            require(raw["first_token"].get(mid) == indexes[0], "HISTORICAL_FIRST_TOKEN")
        sentence = sentence_of.get(indexes[0])
        if "sentence_of" in raw:
            require(
                raw["sentence_of"].get(mid) == sentence, "HISTORICAL_SENTENCE_MISMATCH"
            )
        mentions[mid] = {
            "tokens": list(indexes),
            "surface": surface,
            "sentence": sentence,
            "segments": [
                {"start": offsets[t][0], "end": offsets[t][1], "quote": tokens[t]}
                for t in indexes
            ],
        }
    did = identifier(raw["id"])
    doc = {
        "id": did,
        "topic": did.split("_")[0],
        "revision": revision,
        "text": " ".join(tokens),
        "sha256": text_hash(" ".join(tokens)),
        "offset_unit": "unicode_codepoints_half_open",
        "tokens": list(tokens),
        "token_offsets": offsets,
        "mentions": mentions,
        "sentences": sentences,
    }
    validate_document(doc)
    if "pair_list" in raw:
        expected = set(itertools.permutations(mentions, 2))
        actual = [tuple(x) for x in raw["pair_list"]]
        require(
            len(actual) == len(set(actual)) and set(actual) == expected,
            "HISTORICAL_CANDIDATE_COVERAGE",
        )
    return doc


def public_bundle(suite: dict, controls: list[dict]) -> dict:
    docs = {}
    for raw in suite["documents"]:
        doc = document_from_raw(raw)
        require(doc["id"] not in docs, "DUPLICATE_DOCUMENT")
        docs[doc["id"]] = doc
    ordered = defaultdict(dict)
    for raw in controls:
        # Only these five fields cross the inference boundary.
        did, a, b = raw["doc"], identifier(raw["a"]), identifier(raw["b"])
        require(
            did in docs
            and a in docs[did]["mentions"]
            and b in docs[did]["mentions"]
            and a != b,
            "SCORE_IDS",
        )
        require((a, b) not in ordered[did], "DUPLICATE_ORDERED_SCORE")
        ordered[did][a, b] = {
            "d1": probability(raw["d1"]),
            "luna": probability(raw["luna"]),
        }
    require(set(ordered) == set(docs), "DOCUMENT_SCORE_COVERAGE")
    scores = {}
    for did, doc in docs.items():
        expected = set(itertools.permutations(doc["mentions"], 2))
        require(set(ordered[did]) == expected, "MISSING_REVERSE_OR_CANDIDATE_SCORE")
        scores[did] = []
        for a, b in sorted(itertools.combinations(sorted(doc["mentions"]), 2)):
            r, q = ordered[did][a, b], ordered[did][b, a]
            # Keep the exact four source terms; no max, threshold fitting or defaults.
            fusion = (r["luna"] + q["luna"] + r["d1"] + q["d1"]) / 4
            scores[did].append(
                {
                    "pair": [a, b],
                    "luna_ab": r["luna"],
                    "luna_ba": q["luna"],
                    "d1_ab": r["d1"],
                    "d1_ba": q["d1"],
                    "fusion": fusion,
                    "luna_zero": r["luna"] == 0 and q["luna"] == 0,
                }
            )
    return {
        "schema": "eci-proof-public/v1",
        "threshold": 0.39,
        "documents": docs,
        "scores": scores,
    }


def load_historical_groups(directory: Path, documents: dict) -> dict:
    """Use exact cached T2 groups, never reconstruct connected components from links.

    Invalid, ambiguous or absent caches make a document UNAVAILABLE. Empty groups
    in a valid cached response are AVAILABLE (different from no response).
    """
    found = defaultdict(list)
    if directory.exists():
        for path in sorted(directory.rglob("predictions.jsonl")):
            for line in path.read_text().splitlines():
                row = json.loads(line)
                did, meta = row.get("id"), row.get("metadata", {})
                if did in documents and "coref_groups" in meta:
                    groups = meta["coref_groups"]
                    # No string->int->string ID coercion and no new global partition.
                    valid = isinstance(groups, list) and all(
                        isinstance(g, list) for g in groups
                    )
                    flat = [m for g in groups for m in g] if valid else []
                    valid = valid and all(
                        isinstance(m, str) and m in documents[did]["mentions"]
                        for m in flat
                    )
                    valid = valid and len(flat) == len(set(flat))
                    found[did].append(
                        {
                            "groups": groups,
                            "valid": valid,
                            "path": str(path),
                            "sha256": text_hash(path.read_text()),
                            "model": meta.get("model"),
                            "reasoning_effort": meta.get("reasoning_effort"),
                        }
                    )
    result = {}
    for did in documents:
        hits = found.get(did, [])
        if not hits:
            result[did] = {"status": "UNAVAILABLE", "reason": "MISSING_T2_CACHE"}
        elif any(not h["valid"] for h in hits):
            result[did] = {
                "status": "UNAVAILABLE",
                "reason": "INVALID_T2_GROUP_IDS_OR_OVERLAP",
            }
        elif len({digest(h["groups"]) for h in hits}) != 1:
            result[did] = {"status": "UNAVAILABLE", "reason": "CONFLICTING_T2_CACHES"}
        else:
            result[did] = {
                "status": "AVAILABLE",
                "groups": hits[0]["groups"],
                "provenance": hits,
            }
    return result


def select_roots(bundle: dict, maximum: int = 120, seed: int = 20261027) -> list[dict]:
    require(type(maximum) is int and 0 < maximum <= 120, "SOURCE_LIMIT")
    queues = {}
    topics = defaultdict(list)
    for did, doc in sorted(bundle["documents"].items()):
        rows = [r for r in bundle["scores"][did] if r["fusion"] >= 0.39]
        rows.sort(key=lambda r: (-r["fusion"], digest([seed, did, r["pair"]])))
        if rows:
            queues[did] = deque(rows)
            topics[doc["topic"]].append(did)
    # One root per topic per round, rotating through its documents.
    wheels = {topic: deque(dids) for topic, dids in sorted(topics.items())}
    out = []
    while wheels and len(out) < maximum:
        for topic in list(wheels):
            wheel = wheels[topic]
            did = wheel.popleft()
            row = queues[did].popleft()
            out.append(
                {
                    "id": digest([did, row["pair"]])[:24],
                    "doc": did,
                    "pair": row["pair"],
                    "score": row["fusion"],
                    "generation": 0,
                }
            )
            if queues[did]:
                wheel.append(did)
            if not wheel:
                del wheels[topic]
            if len(out) == maximum:
                break
    return out


def score_map(bundle: dict, did: str) -> dict:
    return {tuple(row["pair"]): row for row in bundle["scores"][did]}


def initial_positive(bundle: dict, did: str) -> set[tuple]:
    return {
        tuple(row["pair"]) for row in bundle["scores"][did] if row["fusion"] >= 0.39
    }
