"""Evaluation enclave: gold is joined only to a sealed complete run.

No import from this module is made by inference, source selection, or requests.
Benchmark correctness and human proof validity are intentionally separate.
"""

from __future__ import annotations

import csv
import itertools
import random
from collections import Counter, defaultdict
from pathlib import Path

from .core import VARIANTS, validate_registry
from .data import initial_positive, score_map
from .pipeline import (
    collect_items,
    judge_requests,
    load_plan,
    read_json,
    read_jsonl,
    write_jsonl_once,
    write_once,
)
from .schema import digest, pair, require


def counts(gold: set, prediction: set) -> dict:
    tp, fp, fn = len(gold & prediction), len(prediction - gold), len(gold - prediction)
    den = 2 * tp + fp + fn
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "f1": 2 * tp / den if den else 0.0,
        "predicted": len(prediction),
        "gold_positive": len(gold),
    }


def sum_counts(rows: list) -> dict:
    total = {
        k: sum(row[k] for row in rows)
        for k in ("tp", "fp", "fn", "predicted", "gold_positive")
    }
    den = 2 * total["tp"] + total["fp"] + total["fn"]
    total["f1"] = 2 * total["tp"] / den if den else 0.0
    return total


def audit_test_s40(pairs_csv: Path, raw_rows: Path) -> dict:
    import json

    raw = {r["id"]: r for r in map(json.loads, raw_rows.read_text().splitlines())}
    rows = list(csv.DictReader(pairs_csv.open()))
    grouped = defaultdict(dict)
    conf = Counter()
    for r in rows:
        require(r["gold"] in {"True", "False"}, "GOLD_BOOLEAN")
        g, p = r["gold"] == "True", float(r["fusion_score"]) >= 0.39
        require(r["fusion_positive"] == str(p), "BASELINE_PREDICTION_PARITY")
        error = ("TP" if g else "FP") if p else ("FN" if g else "TN")
        require(r["error"] == error, "BASELINE_COUNTS_PARITY")
        k = pair(r["a"], r["b"])
        require(k not in grouped[r["doc"]], "DUPLICATE_CANDIDATE")
        grouped[r["doc"]][k] = g
        conf[error] += 1
    require(set(raw) == set(grouped), "BASELINE_DOCUMENT_COVERAGE")
    for did, doc in raw.items():
        pairs = {
            pair(doc["mentions"][a], doc["mentions"][b])
            for edges in doc["relations"].values()
            for a, b in edges
        }
        require(
            pairs == {p for p, g in grouped[did].items() if g}, "BASELINE_GOLD_PARITY"
        )
        require(
            len(grouped[did]) == len(doc["mentions"]) * (len(doc["mentions"]) - 1) // 2,
            "BASELINE_CANDIDATES",
        )
    require(
        (conf["TP"], conf["FP"], conf["FN"]) == (464, 473, 507),
        "TEST_S40_REFERENCE_MISMATCH",
    )
    return {
        "schema": "eci-proof-baseline-audit/v1",
        "counts": dict(conf),
        "documents": len(raw),
        "unordered_candidates": len(rows),
        "f1": 2 * conf["TP"] / (2 * conf["TP"] + conf["FP"] + conf["FN"]),
        "zero_luna_FN": sum(
            r["error"] == "FN" and float(r["luna_score"]) == 0 for r in rows
        ),
        "confident_FP_ge_0_7": sum(
            r["error"] == "FP" and float(r["fusion_score"]) >= 0.7 for r in rows
        ),
        "input_hashes": {str(p): digest(p.read_text()) for p in (pairs_csv, raw_rows)},
        "model_calls": 0,
    }


def export_gold_from_arrow(paths: list[Path], document_ids: set[str]) -> dict:
    """Explicit evaluation-only operation, never called during prepare/request generation."""
    import pyarrow as pa

    out = {}
    for path in paths:
        for row in pa.ipc.open_stream(path).read_all().to_pylist():
            if row["id"] not in document_ids:
                continue
            require(row["id"] not in out, "DUPLICATE_GOLD_DOCUMENT")
            positives = {
                pair(row["mentions"][a], row["mentions"][b])
                for edges in row["relations"].values()
                for a, b in edges
            }
            out[row["id"]] = {
                "positive_pairs": sorted([list(p) for p in positives]),
                "coreference": [
                    [row["mentions"][i] for i in c] for c in row["coref_clusters"]
                ],
            }
    require(set(out) == document_ids, "GOLD_DOCUMENT_COVERAGE")
    return {
        "schema": "eci-proof-evaluation-only/v1",
        "documents": out,
        "warning": "NEVER_INCLUDE_IN_INFERENCE_INPUTS",
    }


def _gold_sets(gold: dict, bundle: dict) -> dict:
    require(
        set(gold["documents"]) == set(bundle["documents"]),
        "EVALUATION_DOCUMENT_COVERAGE",
    )
    out = {}
    for did, r in gold["documents"].items():
        out[did] = {pair(*p) for p in r["positive_pairs"]}
        require(len(out[did]) == len(r["positive_pairs"]), "DUPLICATE_GOLD_PAIR")
        require(out[did] <= set(score_map(bundle, did)), "GOLD_OUTSIDE_CANDIDATES")
    return out


def _identity_error(gold_doc: dict, p: dict):
    if "coreference" not in gold_doc:
        return None
    cid = {m: f"c{i}" for i, g in enumerate(gold_doc["coreference"]) for m in g}
    return cid.get(p["cause"], "s:" + p["cause"]) != cid.get(
        p["alternative"], "s:" + p["alternative"]
    )


def _paired(
    variant_counts: dict,
    docs: list,
    topics: dict,
    left: str,
    right: str,
    samples: int = 2000,
) -> dict:
    keys = sorted({topics[d] for d in docs})
    aggregate = {
        v: {
            t: sum_counts([variant_counts[v][d] for d in docs if topics[d] == t])
            for t in keys
        }
        for v in (left, right)
    }
    point = (
        sum_counts([variant_counts[left][d] for d in docs])["f1"]
        - sum_counts([variant_counts[right][d] for d in docs])["f1"]
    )
    rng = random.Random(20261027)
    deltas = []
    for _ in range(samples):
        draw = rng.choices(keys, k=len(keys)) if keys else []
        deltas.append(
            sum_counts([aggregate[left][t] for t in draw])["f1"]
            - sum_counts([aggregate[right][t] for t in draw])["f1"]
        )
    deltas.sort()
    return {
        "left": left,
        "right": right,
        "documents": len(docs),
        "topics": len(keys),
        "delta_f1": point,
        "topic_bootstrap_ci95": [
            deltas[int(0.025 * samples)],
            deltas[min(samples - 1, int(0.975 * samples))],
        ],
        "resamples": samples,
        "seed": 20261027,
        "development_not_confirmation": True,
    }


def evaluate(
    plan: Path,
    registry_dir: Path,
    run_dir: Path,
    gold: dict,
    output: Path,
    human_audit: dict | None = None,
    diagnostic_receipts: list | None = None,
) -> dict:
    manifest, bundle, roots, groups, config = load_plan(plan)
    registry, run = (
        read_json(registry_dir / "registry.json"),
        read_json(run_dir / "run.json"),
    )
    decisions, usage = (
        read_jsonl(run_dir / "decisions.jsonl"),
        read_jsonl(run_dir / "usage.jsonl"),
    )
    require(
        run["complete"] is True
        and run["plan_hash"] == digest(manifest)
        and run["registry_hash"] == digest(registry)
        and run["decisions_hash"] == digest(decisions)
        and run["usage_hash"] == digest(usage),
        "UNSEALED_OR_INCOMPLETE_RUN",
    )
    validate_registry(bundle, roots, registry)
    diagnostic = None
    if diagnostic_receipts is not None:
        from .diagnostics import inspect_diagnostics

        diagnostic, diagnostic_usage = inspect_diagnostics(
            plan, registry, diagnostic_receipts
        )
        usage = usage + diagnostic_usage
    require(
        not any(d["verdict"] == "PENDING" for d in decisions), "PENDING_IS_NOT_NEGATIVE"
    )
    # The first access to gold is below this seal check.
    gold_sets = _gold_sets(gold, bundle)
    proposals = {p["id"]: p for p in registry["proposals"]}
    root_map = {r["id"]: r for r in roots}
    topics = {d: x["topic"] for d, x in bundle["documents"].items()}
    baseline = {
        d: counts(gold_sets[d], initial_positive(bundle, d))
        for d in bundle["documents"]
    }
    full_baseline = sum_counts(list(baseline.values()))
    common_docs = [d for d in bundle["documents"] if groups[d]["status"] == "AVAILABLE"]
    result, per_doc_counts, per_topic = {}, {}, {}
    for variant in VARIANTS:
        docs = common_docs if variant == "A3" else list(bundle["documents"])
        ds = [d for d in decisions if d["variant"] == variant and d["doc"] in docs]
        added = {
            d: {tuple(r["target"]) for r in ds if r["doc"] == d and r["selected"]}
            for d in docs
        }
        rows = [r for r in ds if r["selected"]]
        tp = sum(len(added[d] & gold_sets[d]) for d in docs)
        n = sum(len(s) for s in added.values())
        base = sum_counts([baseline[d] for d in docs])
        per_doc_counts[variant] = {
            d: counts(gold_sets[d], initial_positive(bundle, d) | added[d])
            for d in docs
        }
        after = sum_counts(list(per_doc_counts[variant].values()))
        zero = sum(
            r["selected"]
            and tuple(r["target"]) in gold_sets[r["doc"]]
            and score_map(bundle, r["doc"])[tuple(r["target"])]["luna_zero"]
            for r in ds
        )
        all_zero_fn = sum(
            row["luna_zero"]
            and tuple(row["pair"]) in gold_sets[d]
            and tuple(row["pair"]) not in initial_positive(bundle, d)
            for d in docs
            for row in bundle["scores"][d]
        )
        identity = [
            _identity_error(gold["documents"][r["doc"]], proposals[r["candidate_id"]])
            for r in rows
        ]
        source_bad = [
            r
            for r in rows
            if tuple(root_map[r["root_id"]]["pair"]) not in gold_sets[r["doc"]]
        ]
        audit_rows = [(human_audit or {}).get(r["candidate_id"]) for r in rows]
        result[variant] = {
            "status": "MEASURED_DEVELOPMENT"
            if run["mode"] == "live"
            else "SYNTHETIC_ONLY",
            "documents": len(docs),
            "selected_source_roots": sum(r["doc"] in docs for r in roots),
            "verified_sources": sum(
                root_map[k]["doc"] in docs and v["status"] == "SUPPORTED"
                for k, v in registry["sources"].items()
            ),
            "candidates": len(ds),
            "additions": n,
            "abstentions": sum(r["verdict"] == "ABSTAIN" for r in ds),
            "unavailable": sum(
                r["variant"] == variant and r["verdict"] == "UNAVAILABLE"
                for r in decisions
            ),
            "tp_added": tp,
            "fp_added": n - tp,
            "marginal_precision": tp / n if n else None,
            "baseline": base,
            "after": after,
            "delta_f1": after["f1"] - base["f1"],
            "neutrality_precision": base["f1"] / 2,
            "U": 2 * tp - base["f1"] * n,
            "zero_vote_tp_recovered": zero,
            "zero_vote_FN_denominator": all_zero_fn,
            "zero_vote_coverage": zero / all_zero_fn if all_zero_fn else None,
            "benchmark_false_source_roots_used": len(
                {r["root_id"] for r in source_bad}
            ),
            "added_from_benchmark_false_sources": len(source_bad),
            "gold_coreference_identity_errors": sum(v is True for v in identity)
            if all(v is not None for v in identity)
            else None,
            "identity_note": "gold-coreference diagnostic, not semantic proof-validity ground truth",
            "source_errors_audited": sum(
                r["source_valid"] == "NO" for r in audit_rows if r
            )
            if any(audit_rows)
            else None,
            "identity_errors_audited": sum(
                r["binding_valid"] == "NO" for r in audit_rows if r
            )
            if any(audit_rows)
            else None,
            "false_additions_from_benchmark_false_sources": sum(
                tuple(r["target"]) not in gold_sets[r["doc"]] for r in source_bad
            ),
            "false_source_expansion": dict(Counter(r["root_id"] for r in source_bad)),
            "scope_errors_audited": sum(
                r["scope_valid"] == "NO" for r in audit_rows if r
            )
            if any(audit_rows)
            else None,
            "proof_validity_audited": {
                "covered_selected": sum(r is not None for r in audit_rows),
                "proof_valid": sum(r["proof_valid"] == "YES" for r in audit_rows if r),
                "unknown_by_dimension": {
                    k: sum(r[k] == "UNKNOWN" for r in audit_rows if r)
                    for k in (
                        "source_valid",
                        "binding_valid",
                        "scope_valid",
                        "proof_valid",
                    )
                },
            },
            "reasons": dict(Counter(r["reason"] for r in ds)),
        }
        common_usage = [u for u in usage if u["stage"] in {"extract", "propose"}]
        own_usage = [u for u in usage if u["stage"] == variant]
        cost = sum(u["cost_usd"] for u in common_usage + own_usage)
        result[variant]["cost"] = {
            "new_calls_common": sum(u["mode"] == "live" for u in common_usage),
            "new_calls_arm": sum(u["mode"] == "live" for u in own_usage),
            "logical_receipts_common": len(common_usage),
            "logical_receipts_arm": len(own_usage),
            "batch_jobs_common": len(
                {u["batch_id"] for u in common_usage if "batch_id" in u}
            ),
            "batch_jobs_arm": len(
                {u["batch_id"] for u in own_usage if "batch_id" in u}
            ),
            "tokens_common": sum(
                u["input_tokens"] + u["output_tokens"] for u in common_usage
            ),
            "tokens_arm": sum(
                u["input_tokens"] + u["output_tokens"] for u in own_usage
            ),
            "usd_common_plus_arm": cost,
            "usd_per_TP": cost / tp if tp else None,
            "historical_cache_new_cost_usd": 0,
            "note": "Common stages attributed per arm; do not sum these arm costs.",
        }
        per_topic[variant] = {}
        for topic in sorted({topics[d] for d in docs}):
            subset = [d for d in docs if topics[d] == topic]
            b = sum_counts([baseline[d] for d in subset])
            a = sum_counts([per_doc_counts[variant][d] for d in subset])
            per_topic[variant][topic] = {
                "documents": len(subset),
                "tp_added": a["tp"] - b["tp"],
                "fp_added": a["fp"] - b["fp"],
                "delta_f1": a["f1"] - b["f1"],
            }
    comparisons = []
    for left, right in itertools.combinations(VARIANTS, 2):
        docs = sorted(set(per_doc_counts[left]) & set(per_doc_counts[right]))
        comparison = _paired(per_doc_counts, docs, topics, left, right)
        comparison["precision_at_predeclared_budgets"] = {}
        for budget in (20, 50):
            selected_by_arm = {
                v: [
                    d
                    for d in decisions
                    if d["variant"] == v and d["doc"] in docs and d["selected"]
                ]
                for v in (left, right)
            }
            if any(len(xs) < budget for xs in selected_by_arm.values()):
                comparison["precision_at_predeclared_budgets"][str(budget)] = None
            else:
                precision = {
                    v: sum(
                        tuple(d["target"]) in gold_sets[d["doc"]] for d in xs[:budget]
                    )
                    / budget
                    for v, xs in selected_by_arm.items()
                }
                comparison["precision_at_predeclared_budgets"][str(budget)] = {
                    "left": precision[left],
                    "right": precision[right],
                    "difference": precision[left] - precision[right],
                }
        comparisons.append(comparison)
    a2 = result["A2"]
    ready = (
        a2["additions"] >= 50
        and a2["tp_added"] >= 20
        and (a2["marginal_precision"] or 0) >= 0.35
        and a2["U"] > 0
    )
    better = all(
        next(
            c["delta_f1"] if c["left"] == "A2" else -c["delta_f1"]
            for c in comparisons
            if {c["left"], c["right"]} == {"A2", v}
        )
        > 0
        for v in ("A0", "A1", "A3")
    )
    audit_complete = bool(human_audit) and all(k in human_audit for k in proposals)
    accepted_a2_ids = [
        d["candidate_id"] for d in decisions if d["variant"] == "A2" and d["selected"]
    ]
    audit_decisive = audit_complete and all(
        all(
            human_audit[cid][k] != "UNKNOWN"
            for k in ("source_valid", "binding_valid", "scope_valid", "proof_valid")
        )
        for cid in accepted_a2_ids
    )
    intervals_positive = all(
        (
            c["topic_bootstrap_ci95"][0] > 0
            if c["left"] == "A2"
            else c["topic_bootstrap_ci95"][1] < 0
        )
        for c in comparisons
        if "A2" in {c["left"], c["right"]}
    )
    within_budget = (
        sum(u["cost_usd"] for u in usage) <= config["max_usd"]
        and sum(u["input_tokens"] + u["output_tokens"] for u in usage)
        <= config["max_billed_tokens"]
    )
    go = (
        ready
        and better
        and intervals_positive
        and audit_complete
        and audit_decisive
        and within_budget
        and run["mode"] == "live"
        and run["expected_version_verified"]
    )
    if (
        diagnostic is None
        or diagnostic["claims_reuse_without_witness"]
        or diagnostic["unscorable_raw_obligation"]
    ):
        go = False
    output_result = {
        "schema": "eci-proof-evaluation/v1",
        "witness_dependence_diagnostic": diagnostic,
        "plan_hash": digest(manifest),
        "run_hash": digest(run),
        "gold_hash": digest(gold),
        "baseline_full": full_baseline,
        "variants": result,
        "per_topic": per_topic,
        "paired": comparisons,
        "total_new_usage": {
            "calls": sum(u["mode"] == "live" for u in usage),
            "logical_receipts": len(usage),
            "batch_jobs": len({u["batch_id"] for u in usage if "batch_id" in u}),
            "call_unit": "logical model requests; batch_jobs are distinct submission jobs",
            "tokens": sum(u["input_tokens"] + u["output_tokens"] for u in usage),
            "usd": sum(u["cost_usd"] for u in usage),
        },
        "recommendation": "GO_TO_INDEPENDENT_CONFIRMATION_PLANNING"
        if go
        else "NO_GO_OR_INDETERMINATE",
        "decision_gates": {
            "volume_and_marginal_utility": ready,
            "better_point_F1_than_all_controls": better,
            "paired_topic_intervals_positive": intervals_positive,
            "human_audit_complete": audit_complete,
            "human_audit_decisive_for_selected_A2": audit_decisive,
            "within_budget": within_budget,
            "expected_version_verified": run["expected_version_verified"],
        },
        "not_confirmed_gain": True,
        "human_audit_complete": audit_complete,
    }
    write_once(output / "comparison.json", output_result)
    write_once(
        output / "abstentions-errors.json",
        {
            "source_gate": registry["sources"],
            "source_gate_evaluation_only": [
                {
                    "root_id": r["id"],
                    "doc": r["doc"],
                    "gold_correct": tuple(r["pair"]) in gold_sets[r["doc"]],
                    "gate_status": registry["sources"][r["id"]]["status"],
                    "reason": registry["sources"][r["id"]].get("reason"),
                }
                for r in roots
            ],
            "candidate_reasons": {v: r["reasons"] for v, r in result.items()},
            "cases_evaluation_only": [
                {
                    **d,
                    "target_gold_correct": tuple(d["target"]) in gold_sets[d["doc"]],
                    "source_gold_correct": tuple(root_map[d["root_id"]]["pair"])
                    in gold_sets[d["doc"]],
                    "identity_gold_error": _identity_error(
                        gold["documents"][d["doc"]], proposals[d["candidate_id"]]
                    ),
                    "human_audit": (human_audit or {}).get(d["candidate_id"]),
                }
                for d in decisions
            ],
        },
    )
    import io

    stream = io.StringIO()
    columns = [
        "variant",
        "status",
        "documents",
        "selected_source_roots",
        "verified_sources",
        "candidates",
        "additions",
        "abstentions",
        "unavailable",
        "tp_added",
        "fp_added",
        "marginal_precision",
        "delta_f1",
        "U",
        "zero_vote_tp_recovered",
        "benchmark_false_source_roots_used",
        "gold_coreference_identity_errors",
        "source_errors_audited",
        "identity_errors_audited",
        "scope_errors_audited",
    ]
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for variant, row in result.items():
        writer.writerow(
            {"variant": variant, **{k: row[k] for k in columns if k != "variant"}}
        )
    csv_path = output / "comparison.csv"
    if csv_path.exists():
        require(csv_path.read_text() == stream.getvalue(), "EVALUATION_CSV_OVERWRITE")
    else:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("x") as f:
            f.write(stream.getvalue())
    return output_result


def blind_audit(
    plan: Path, registry: dict, output: Path, judge_receipts: list | None = None
) -> None:
    _manifest, bundle, roots, _, _ = load_plan(plan)
    validate_registry(bundle, roots, registry)
    transport_outputs = {}
    if judge_receipts is not None:
        requests = [r for r in judge_requests(plan, registry) if r["stage"] == "A2"]
        wanted = {r["id"] for r in requests}
        transport_outputs, _ = collect_items(
            requests,
            [r for r in judge_receipts if r["request_id"] in wanted],
            "candidate_id",
        )
    packets, mapping = [], {}
    for p in registry["proposals"]:
        opaque = digest([digest(registry), "blind_audit", p["id"]])[:24]
        mapping[opaque] = p["id"]
        packets.append(
            {
                "audit_id": opaque,
                "document": bundle["documents"][p["doc"]],
                "cause": p["cause"],
                "alternative": p["alternative"],
                "effect": p["effect"],
                "source_certificate": registry["sources"][p["root_id"]]["certificate"],
                "proposed_transport": {
                    k: v
                    for k, v in transport_outputs.get(p["id"], {}).items()
                    if k != "candidate_id"
                }
                or None,
                "questions": {
                    k: "YES|NO|UNKNOWN"
                    for k in (
                        "source_valid",
                        "binding_valid",
                        "scope_valid",
                        "proof_valid",
                    )
                },
                "instruction": "Judge textual validity. No benchmark label, source score, arm, or acceptance is shown.",
            }
        )
    random.Random(20261027).shuffle(packets)
    write_jsonl_once(output / "blind-packets.jsonl", packets)
    write_once(output / "private-audit-id-map.json", mapping)
