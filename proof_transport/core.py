"""Deterministic gates for A0/A1/A2/A3 and one-generation bounded selection.

Semantic judgments are frozen-model outputs, not mathematical proofs. This code
checks the structural obligations and refuses missing/unknown/ambiguous evidence.
"""

from __future__ import annotations

from collections import Counter

from .data import initial_positive, score_map
from .schema import (
    FACTS,
    Invalid,
    citations,
    compare_fact,
    compare_scope,
    digest,
    pair,
    require,
    validate_all_citations,
    validate_certificate,
    validate_frame,
    validate_root,
    validate_scope,
)

VARIANTS = ("A0", "A1", "A2", "A3")


def build_registry(
    bundle: dict, roots: list[dict], extractions: dict, verifications: dict
) -> dict:
    require(
        len(roots) <= 120 and len({r["id"] for r in roots}) == len(roots),
        "ROOT_LIMIT_OR_DUPLICATE",
    )
    require(set(extractions) == {r["id"] for r in roots}, "INCOMPLETE_EXTRACTION")
    sources, proposals, rejections = {}, [], []
    for root in roots:
        validate_root(root, bundle)
        doc, rid = bundle["documents"][root["doc"]], root["id"]
        item = extractions[rid]
        try:
            require(
                set(item) == {"source_id", "verdict", "reason", "certificate"}
                and item["source_id"] == rid,
                "EXTRACTION_SCHEMA",
            )
            require(
                item["verdict"] in {"SUPPORTED", "ABSTAIN", "REJECT"},
                "EXTRACTION_VERDICT",
            )
            require(item["verdict"] == "SUPPORTED", "SOURCE_EXTRACTION_ABSTAIN")
            cert = item["certificate"]
            validate_certificate(cert, root, doc)
            check = verifications.get(rid)
            require(
                isinstance(check, dict)
                and set(check)
                == {
                    "source_id",
                    "source_certificate_hash",
                    "verdict",
                    "reason",
                    "alternative_ids",
                    "justification",
                    "citations",
                },
                "VERIFICATION_SCHEMA",
            )
            require(
                check["source_id"] == rid
                and check["source_certificate_hash"] == digest(cert),
                "STALE_VERIFICATION",
            )
            validate_all_citations(check, doc)
            require(
                check["verdict"] in {"SUPPORTED", "ABSTAIN", "REJECT"},
                "VERIFICATION_VERDICT",
            )
            require(check["verdict"] == "SUPPORTED", "SOURCE_VERIFIER_ABSTAIN")
            citations(check["citations"], doc)
            require(
                isinstance(check["justification"], str)
                and bool(check["justification"].strip()),
                "MISSING_JUSTIFICATION",
            )
            alternatives = check["alternative_ids"]
            require(
                isinstance(alternatives, list)
                and len(alternatives) <= 2
                and all(isinstance(x, str) for x in alternatives)
                and len(alternatives) == len(set(alternatives)),
                "PROPOSAL_CAP_OR_DUPLICATE",
            )
            require(
                all(
                    x in doc["mentions"] and x not in {cert["cause"], cert["effect"]}
                    for x in alternatives
                ),
                "ALTERNATIVE_IDS",
            )
            sources[rid] = {
                "status": "SUPPORTED",
                "certificate": cert,
                "verification": check,
            }
            scores = score_map(bundle, root["doc"])
            for alternative in alternatives:
                target = list(pair(alternative, cert["effect"]))
                if tuple(target) not in scores or tuple(target) in initial_positive(
                    bundle, root["doc"]
                ):
                    rejections.append(
                        {
                            "source_id": rid,
                            "alternative": alternative,
                            "reason": "INVALID_OR_INITIAL_POSITIVE",
                        }
                    )
                    continue
                p = {
                    "root_id": rid,
                    "doc": root["doc"],
                    "cause": cert["cause"],
                    "effect": cert["effect"],
                    "alternative": alternative,
                    "target": target,
                    "root_score": root["score"],
                    "source_certificate_hash": digest(cert),
                    "generation": 1,
                    "reusable_as_source": False,
                }
                p["id"] = digest(p)[:24]
                proposals.append(p)
        except (Invalid, KeyError, TypeError) as error:
            sources[rid] = {"status": "ABSTAIN", "reason": str(error)}
    require(set(verifications) <= set(extractions), "EXTRA_VERIFICATION_IDS")
    return {
        "schema": "eci-proof-registry/v1",
        "bundle_hash": digest(bundle),
        "roots_hash": digest(roots),
        "sources": sources,
        "proposals": proposals,
        "proposal_rejections": rejections,
        "extractions": extractions,
        "verifications": verifications,
    }


def validate_registry(bundle: dict, roots: list[dict], registry: dict) -> None:
    rebuilt = build_registry(
        bundle, roots, registry["extractions"], registry["verifications"]
    )
    require(registry == rebuilt, "STALE_OR_TAMPERED_REGISTRY")
    require(len(registry["proposals"]) <= 240, "TOTAL_PROPOSAL_CAP")
    require(
        len({p["id"] for p in registry["proposals"]}) == len(registry["proposals"]),
        "DUPLICATE_PROPOSAL_ID",
    )


def _gate_a1(raw: dict, cert: dict, doc: dict) -> tuple[str, str]:
    require(
        set(raw)
        == {
            "candidate_id",
            "alternative_frame",
            "discriminating_fields",
            "justification",
        },
        "A1_SCHEMA",
    )
    frame = raw["alternative_frame"]
    validate_frame(frame, doc)
    source = cert["frames"][cert["cause"]]
    scope = compare_scope(source["scope"], frame["scope"])
    if scope != "AGREEMENT":
        return "ABSTAIN", "SCOPE_" + scope
    fields = raw["discriminating_fields"]
    require(
        isinstance(fields, list)
        and len(fields) == len(set(fields))
        and set(fields) <= set(FACTS),
        "DISCRIMINANTS",
    )
    comparisons = {f: compare_fact(source[f], frame[f]) for f in FACTS}
    if "CONFLICT" in comparisons.values():
        return "ABSTAIN", "FACT_CONFLICT"
    if any(comparisons[f] == "UNKNOWN" for f in fields):
        return "ABSTAIN", "UNKNOWN_DISCRIMINANT"
    observed = [
        f
        for f in fields
        if comparisons[f] == "AGREEMENT"
        and source[f]["state"] == frame[f]["state"] == "OBSERVED"
    ]
    return (
        ("ACCEPT", "COMPATIBLE_ONLY")
        if len(observed) >= 2
        else ("ABSTAIN", "INSUFFICIENT_OBSERVED_AGREEMENTS")
    )


def _gate_a2(raw: dict, cert: dict, doc: dict, proposal: dict) -> tuple[str, str]:
    require(
        set(raw)
        == {
            "candidate_id",
            "source_certificate_hash",
            "evidence_hash",
            "binding_kind",
            "binding",
            "alternative_frame",
            "relation_scope",
            "discriminating_fields",
            "obligations",
            "competitors",
            "new_causal_premises",
            "justification",
        },
        "A2_SCHEMA",
    )
    require(
        raw["source_certificate_hash"] == digest(cert)
        and raw["evidence_hash"] == digest(cert["evidence"]),
        "EVIDENCE_CHANGED",
    )
    require(
        raw["binding_kind"]
        in {"EXPLICIT_ANAPHORA", "UNIQUE_INCIDENT", "OCCURRENCE_IDENTIFICATION"},
        "NO_POSITIVE_BINDING",
    )
    binding = citations(raw["binding"], doc)
    for segment in doc["mentions"][proposal["alternative"]]["segments"]:
        require(
            any(
                s.start <= segment["start"] and segment["end"] <= s.end for s in binding
            ),
            "ALTERNATIVE_NOT_IN_BINDING_EVIDENCE",
        )
    validate_frame(raw["alternative_frame"], doc)
    validate_scope(raw["relation_scope"], doc)
    if compare_scope(cert["relation_scope"], raw["relation_scope"]) != "AGREEMENT":
        return "ABSTAIN", "RELATION_SCOPE_CHANGED"
    source = cert["frames"][cert["cause"]]
    if compare_scope(source["scope"], raw["alternative_frame"]["scope"]) != "AGREEMENT":
        return "ABSTAIN", "SCOPE_INCOMPATIBLE_OR_UNKNOWN"
    fields = raw["discriminating_fields"]
    require(
        isinstance(fields, list)
        and len(fields) == len(set(fields))
        and set(fields) <= set(FACTS),
        "DISCRIMINANTS",
    )
    for field in FACTS:
        state = compare_fact(source[field], raw["alternative_frame"][field])
        if state == "CONFLICT" or (state == "UNKNOWN" and field in fields):
            return "ABSTAIN", "FACT_CONFLICT_OR_UNKNOWN_DISCRIMINANT"
    require(raw["new_causal_premises"] is False, "NEW_EXTRACTION_NOT_TRANSPORT")
    obligations = raw["obligations"]
    require(
        isinstance(obligations, dict)
        and set(obligations)
        == {"binding", "causal_reuse", "granularity", "scope", "discriminants"},
        "OBLIGATION_SCHEMA",
    )
    require(
        all(v in {"SUPPORTED", "CONFLICT", "UNKNOWN"} for v in obligations.values()),
        "OBLIGATION_VALUE",
    )
    if any(v != "SUPPORTED" for v in obligations.values()):
        return "ABSTAIN", "UNSUPPORTED_OBLIGATION"
    rivals = raw["competitors"]
    require(
        isinstance(rivals, dict)
        and set(rivals)
        == {"covered_mentions", "covered_sentences", "residual", "rivals"},
        "COMPETITOR_SCHEMA",
    )
    require(
        isinstance(rivals["covered_mentions"], list)
        and len(rivals["covered_mentions"]) == len(doc["mentions"])
        and set(rivals["covered_mentions"]) == set(doc["mentions"]),
        "INCOMPLETE_COMPETITOR_MENTION_SCAN",
    )
    require(
        rivals["covered_sentences"] == list(range(len(doc["sentences"]))),
        "INCOMPLETE_COMPETITOR_SENTENCE_SCAN",
    )
    require(
        rivals["residual"] in {"NONE", "UNKNOWN"}
        and isinstance(rivals["rivals"], list),
        "COMPETITOR_VALUES",
    )
    if rivals["residual"] != "NONE":
        return "ABSTAIN", "UNRESOLVED_COMPETITOR"
    for rival in rivals["rivals"]:
        require(
            isinstance(rival, dict)
            and set(rival) == {"mention_id", "citations", "outcome", "reason"},
            "RIVAL_SCHEMA",
        )
        require(
            rival["mention_id"] is None or rival["mention_id"] in doc["mentions"],
            "RIVAL_ID",
        )
        citations(rival["citations"], doc)
        require(
            rival["outcome"] in {"EXCLUDED", "PLAUSIBLE", "UNKNOWN"}, "RIVAL_OUTCOME"
        )
        require(
            isinstance(rival["reason"], str) and bool(rival["reason"].strip()),
            "RIVAL_REASON",
        )
        if rival["outcome"] != "EXCLUDED":
            return "ABSTAIN", "UNRESOLVED_COMPETITOR"
    return "ACCEPT", "SAME_WITNESS_SUPPORTED"


def decide(
    variant: str,
    proposal: dict,
    registry: dict,
    bundle: dict,
    raw: dict | None,
    groups: dict,
) -> dict:
    require(variant in VARIANTS, "UNKNOWN_VARIANT")
    result = {
        "variant": variant,
        "candidate_id": proposal["id"],
        "root_id": proposal["root_id"],
        "doc": proposal["doc"],
        "target": proposal["target"],
        "verdict": "ABSTAIN",
        "reason": "PENDING_MODEL_RESPONSE",
        "raw_hash": digest(raw) if raw is not None else None,
        "selected": False,
    }
    cert = registry["sources"][proposal["root_id"]]["certificate"]
    doc = bundle["documents"][proposal["doc"]]
    try:
        if variant == "A3":
            cached = groups[proposal["doc"]]
            if cached["status"] != "AVAILABLE":
                result.update(verdict="UNAVAILABLE", reason=cached["reason"])
            else:
                # Historical T2 membership; no new chains, no in-group suppression.
                membership = {
                    m: f"g{i}" for i, g in enumerate(cached["groups"]) for m in g
                }
                group = lambda m: membership.get(m, "singleton:" + m)
                same = group(proposal["cause"]) == group(
                    proposal["alternative"]
                ) and group(proposal["cause"]) != group(proposal["effect"])
                result.update(
                    verdict="ACCEPT" if same else "ABSTAIN",
                    reason="T2_PROJECTED_COMMON_ROOT" if same else "T2_NO_MATCH",
                )
        elif raw is None:
            result.update(verdict="PENDING")
        else:
            require(
                isinstance(raw, dict) and raw.get("candidate_id") == proposal["id"],
                "DECISION_ID",
            )
            validate_all_citations(raw, doc)
            require(
                isinstance(raw.get("justification"), str)
                and bool(raw["justification"].strip()),
                "MISSING_JUSTIFICATION",
            )
            if variant == "A0":
                require(
                    set(raw) == {"candidate_id", "identity", "justification"},
                    "A0_SCHEMA",
                )
                require(
                    raw["identity"] in {"SAME", "DIFFERENT", "UNKNOWN"},
                    "IDENTITY_VALUE",
                )
                verdict = "ACCEPT" if raw["identity"] == "SAME" else "ABSTAIN"
                reason = (
                    "IDENTITY_ONLY"
                    if verdict == "ACCEPT"
                    else "IDENTITY_" + raw["identity"]
                )
            elif variant == "A1":
                verdict, reason = _gate_a1(raw, cert, doc)
            else:
                verdict, reason = _gate_a2(raw, cert, doc, proposal)
            result.update(verdict=verdict, reason=reason)
    except (Invalid, KeyError, TypeError) as error:
        result.update(verdict="ABSTAIN", reason="INVALID_OR_INSUFFICIENT:" + str(error))
    return result


def select_additions(decisions: list[dict], registry: dict, bundle: dict) -> list[dict]:
    """Same ordering and limits for every arm. Never change baseline P0 or recurse."""
    proposals = {p["id"]: p for p in registry["proposals"]}
    require(
        len({(d["variant"], d["candidate_id"]) for d in decisions}) == len(decisions),
        "DUPLICATE_DECISION",
    )
    expected = {(v, p) for v in VARIANTS for p in proposals}
    require(
        {(d["variant"], d["candidate_id"]) for d in decisions} == expected,
        "DECISION_COVERAGE",
    )
    output = []
    for variant in VARIANTS:
        per_source, per_doc, used = Counter(), Counter(), set()
        rows = [dict(d) for d in decisions if d["variant"] == variant]
        rows.sort(
            key=lambda d: (
                -proposals[d["candidate_id"]]["root_score"],
                d["candidate_id"],
            )
        )
        for d in rows:
            p = proposals[d["candidate_id"]]
            require(
                d["target"] == p["target"]
                and d["doc"] == p["doc"]
                and d["root_id"] == p["root_id"],
                "DECISION_ENDPOINTS_CHANGED",
            )
            require(
                type(p["generation"]) is int
                and p["generation"] == 1
                and p["reusable_as_source"] is False,
                "RECURSION",
            )
            doc = bundle["documents"][p["doc"]]
            require(
                all(
                    x in doc["mentions"]
                    for x in (p["cause"], p["effect"], p["alternative"])
                )
                and p["alternative"] not in {p["cause"], p["effect"]}
                and p["target"] == list(pair(p["alternative"], p["effect"]))
                and pair(p["cause"], p["effect"]) in initial_positive(bundle, p["doc"])
                and tuple(p["target"]) in score_map(bundle, p["doc"]),
                "INVALID_TRANSPORT_ENDPOINTS",
            )
            require(d["selected"] is False, "ALREADY_SELECTED")
            if d["verdict"] == "ACCEPT":
                key = (d["doc"], tuple(d["target"]))
                reason = None
                if tuple(d["target"]) in initial_positive(bundle, d["doc"]):
                    reason = "INITIAL_POSITIVE"
                elif key in used:
                    reason = "DUPLICATE_TARGET"
                elif per_source[d["root_id"]] >= 1:
                    reason = "SOURCE_CAP"
                elif per_doc[d["doc"]] >= 5:
                    reason = "DOCUMENT_CAP"
                if reason:
                    d.update(verdict="ABSTAIN", reason=reason)
                else:
                    per_source[d["root_id"]] += 1
                    per_doc[d["doc"]] += 1
                    used.add(key)
                    d["selected"] = True
            output.append(d)
        require(
            max(per_source.values(), default=0) <= 1
            and max(per_doc.values(), default=0) <= 5,
            "CAP_VIOLATION",
        )
    return output
