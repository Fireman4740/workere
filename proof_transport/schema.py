"""Strict, label-free public types and exact citation validation.

Offsets are Unicode code points in an immutable ``' '.join(tokens)`` text.
No fuzzy matching, coercion of booleans to offsets, learned rules, or gold input.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any


class Invalid(ValueError):
    """Invalid input or an unmet structural proof obligation."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Invalid(message)


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def identifier(value: Any) -> str:
    require(
        isinstance(value, str) and bool(value) and value == value.strip(), "INVALID_ID"
    )
    return value


def pair(a: str, b: str) -> tuple[str, str]:
    identifier(a)
    identifier(b)
    require(a != b, "SELF_PAIR")
    return tuple(sorted((a, b)))


def probability(value: Any) -> float:
    require(
        type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
        "INVALID_SCORE",
    )
    return float(value)


@dataclass(frozen=True)
class Citation:
    start: int
    end: int
    quote: str

    @classmethod
    def parse(cls, raw: dict, document: dict) -> Citation:
        require(
            isinstance(raw, dict) and set(raw) == {"start", "end", "quote"},
            "CITATION_SCHEMA",
        )
        start, end, quote = raw["start"], raw["end"], raw["quote"]
        require(
            type(start) is int and type(end) is int and isinstance(quote, str),
            "CITATION_TYPES",
        )
        require(0 <= start < end <= len(document["text"]), "CITATION_BOUNDS")
        require(document["text"][start:end] == quote, "CITATION_MISMATCH")
        return cls(start, end, quote)


def citations(raw: Any, doc: dict, *, nonempty: bool = True) -> list[Citation]:
    require(isinstance(raw, list) and (bool(raw) or not nonempty), "MISSING_CITATION")
    return [Citation.parse(x, doc) for x in raw]


def validate_all_citations(value: Any, doc: dict) -> None:
    """Check every declared span, not just the first supporting passage."""
    if isinstance(value, dict):
        if {"start", "end", "quote"} & value.keys():
            Citation.parse(value, doc)
        else:
            for v in value.values():
                validate_all_citations(v, doc)
    elif isinstance(value, list):
        for v in value:
            validate_all_citations(v, doc)


FACTS = ("participants", "incident", "time", "place", "granularity")
STATES = {"OBSERVED", "INFERRED", "UNKNOWN", "CONTRADICTED"}
SCOPE_KEYS = {
    "state",
    "holder",
    "polarity",
    "world",
    "modality",
    "operators",
    "citations",
}


def validate_fact(value: dict, doc: dict, field: str | None = None) -> None:
    require(
        isinstance(value, dict) and set(value) == {"state", "value", "citations"},
        "FACT_SCHEMA",
    )
    require(value["state"] in STATES, "FACT_STATE")
    if value["state"] == "UNKNOWN":
        require(
            value["value"] is None and value["citations"] == [],
            "UNKNOWN_IS_NOT_AGREEMENT",
        )
    else:
        if field == "participants":
            roles = value["value"]
            require(
                isinstance(roles, dict)
                and bool(roles)
                and set(roles)
                <= {
                    "agent",
                    "patient",
                    "instrument",
                    "affected_object",
                    "beneficiary",
                    "other",
                }
                and all(
                    isinstance(xs, list)
                    and bool(xs)
                    and all(isinstance(x, str) and x.strip() for x in xs)
                    and len(xs) == len(set(xs))
                    for xs in roles.values()
                ),
                "PARTICIPANT_ROLES",
            )
        else:
            require(
                isinstance(value["value"], str) and bool(value["value"].strip()),
                "FACT_VALUE",
            )
        citations(value["citations"], doc)


def validate_scope(value: dict, doc: dict) -> None:
    require(isinstance(value, dict) and set(value) == SCOPE_KEYS, "SCOPE_SCHEMA")
    require(value["state"] in STATES, "SCOPE_STATE")
    require(
        isinstance(value["operators"], list)
        and all(isinstance(x, str) for x in value["operators"]),
        "SCOPE_OPERATORS",
    )
    if value["state"] == "UNKNOWN":
        require(
            all(value[k] is None for k in ("holder", "polarity", "world", "modality"))
            and value["operators"] == []
            and value["citations"] == [],
            "UNKNOWN_SCOPE",
        )
    else:
        require(
            all(
                isinstance(value[k], str) and bool(value[k])
                for k in ("holder", "polarity", "world", "modality")
            ),
            "SCOPE_VALUE",
        )
        citations(value["citations"], doc)


def validate_frame(frame: dict, doc: dict) -> None:
    require(
        isinstance(frame, dict) and set(frame) == set(FACTS) | {"scope"}, "FRAME_SCHEMA"
    )
    for key in FACTS:
        validate_fact(frame[key], doc, key)
    validate_scope(frame["scope"], doc)


def compare_fact(left: dict, right: dict) -> str:
    if "CONTRADICTED" in (left["state"], right["state"]):
        return "CONFLICT"
    if "UNKNOWN" in (left["state"], right["state"]):
        return "UNKNOWN"
    return "AGREEMENT" if left["value"] == right["value"] else "CONFLICT"


def compare_scope(left: dict, right: dict) -> str:
    if "CONTRADICTED" in (left["state"], right["state"]):
        return "CONFLICT"
    if "UNKNOWN" in (left["state"], right["state"]):
        return "UNKNOWN"
    keys = ("holder", "polarity", "world", "modality", "operators")
    return "AGREEMENT" if all(left[k] == right[k] for k in keys) else "CONFLICT"


def validate_document(doc: dict) -> None:
    require(
        set(doc)
        == {
            "id",
            "topic",
            "revision",
            "text",
            "sha256",
            "offset_unit",
            "tokens",
            "token_offsets",
            "mentions",
            "sentences",
        },
        "PUBLIC_DOCUMENT_SCHEMA",
    )
    identifier(doc["id"])
    identifier(doc["topic"])
    require(doc["offset_unit"] == "unicode_codepoints_half_open", "OFFSET_CONVENTION")
    require(doc["text"] == " ".join(doc["tokens"]), "CANONICAL_TEXT")
    require(doc["sha256"] == text_hash(doc["text"]), "STALE_DOCUMENT_HASH")
    require(len(doc["tokens"]) == len(doc["token_offsets"]), "TOKEN_OFFSETS")
    position = 0
    for token, (start, end) in zip(doc["tokens"], doc["token_offsets"]):
        require(
            type(start) is int
            and type(end) is int
            and start == position
            and end == start + len(token)
            and doc["text"][start:end] == token,
            "TOKEN_OFFSET",
        )
        position = end + 1
    sentence_of = {}
    for i, bounds in enumerate(doc["sentences"]):
        require(
            len(bounds) == 2 and all(type(t) is int for t in bounds), "SENTENCE_TYPES"
        )
        start, end = bounds
        require(0 <= start < end <= len(doc["tokens"]), "SENTENCE_BOUNDS")
        for t in range(start, end):
            require(t not in sentence_of, "OVERLAPPING_SENTENCES")
            sentence_of[t] = i
    require(set(sentence_of) == set(range(len(doc["tokens"]))), "SENTENCE_COVERAGE")
    require(
        isinstance(doc["mentions"], dict) and bool(doc["mentions"]), "EMPTY_MENTIONS"
    )
    for mid, m in doc["mentions"].items():
        identifier(mid)
        require(
            set(m) == {"tokens", "surface", "segments", "sentence"}, "MENTION_SCHEMA"
        )
        indexes = m["tokens"]
        require(
            isinstance(indexes, list)
            and bool(indexes)
            and all(type(t) is int and 0 <= t < len(doc["tokens"]) for t in indexes)
            and indexes == sorted(set(indexes)),
            "MENTION_TOKENS",
        )
        require(
            m["surface"] == " ".join(doc["tokens"][t] for t in indexes),
            "MENTION_SURFACE",
        )
        expected = [
            {
                "start": doc["token_offsets"][t][0],
                "end": doc["token_offsets"][t][1],
                "quote": doc["tokens"][t],
            }
            for t in indexes
        ]
        require(m["segments"] == expected, "MENTION_SEGMENTS")
        validate_all_citations(m["segments"], doc)
        require(
            all(sentence_of[t] == m["sentence"] for t in indexes), "MENTION_SENTENCE"
        )


def validate_root(root: dict, bundle: dict) -> None:
    require(set(root) == {"id", "doc", "pair", "score", "generation"}, "ROOT_SCHEMA")
    require(
        type(root["generation"]) is int and root["generation"] == 0, "RECURSIVE_ROOT"
    )
    doc = bundle["documents"][root["doc"]]
    a, b = root["pair"]
    require(a in doc["mentions"] and b in doc["mentions"], "UNKNOWN_MENTION")
    require(list(pair(a, b)) == root["pair"], "NONCANONICAL_ROOT")
    row = next(
        (r for r in bundle["scores"][root["doc"]] if r["pair"] == root["pair"]), None
    )
    require(
        row is not None and row["fusion"] == root["score"] >= 0.39,
        "ROOT_NOT_INITIAL_POSITIVE",
    )
    require(root["id"] == digest([root["doc"], root["pair"]])[:24], "ROOT_ID")


def validate_certificate(cert: dict, root: dict, doc: dict) -> None:
    require(
        set(cert)
        == {
            "source_id",
            "document_sha256",
            "cause",
            "effect",
            "evidence",
            "frames",
            "relation_scope",
            "support_mode",
            "justification",
        },
        "CERTIFICATE_SCHEMA",
    )
    require(
        cert["source_id"] == root["id"] and cert["document_sha256"] == doc["sha256"],
        "STALE_SOURCE",
    )
    require(
        {cert["cause"], cert["effect"]} == set(root["pair"])
        and cert["cause"] != cert["effect"],
        "SOURCE_ENDPOINTS",
    )
    require(cert["support_mode"] == "EXPLICIT", "INSUFFICIENT_SOURCE_SUPPORT")
    evidence = citations(cert["evidence"], doc)
    require(
        isinstance(cert["justification"], str) and bool(cert["justification"].strip()),
        "MISSING_JUSTIFICATION",
    )
    require(set(cert["frames"]) == {cert["cause"], cert["effect"]}, "SOURCE_FRAMES")
    for frame in cert["frames"].values():
        validate_frame(frame, doc)
        require(
            frame["scope"]["state"] in {"OBSERVED", "INFERRED"},
            "UNKNOWN_SOURCE_EVENT_SCOPE",
        )
    validate_scope(cert["relation_scope"], doc)
    require(
        cert["relation_scope"]["state"] in {"OBSERVED", "INFERRED"},
        "UNKNOWN_SOURCE_SCOPE",
    )
    for mid in (cert["cause"], cert["effect"]):
        for segment in doc["mentions"][mid]["segments"]:
            require(
                any(
                    e.start <= segment["start"] and segment["end"] <= e.end
                    for e in evidence
                ),
                "SOURCE_MENTION_NOT_IN_EVIDENCE",
            )
    validate_all_citations(cert, doc)
