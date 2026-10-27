"""Staged offline-first runner: prepare -> extract -> propose -> freeze -> judge.

All artifacts are write-once. A candidate registry is never invented before the
common source verification and retrieval responses exist. Gold is not imported.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from . import __version__
from .core import VARIANTS, build_registry, decide, select_additions, validate_registry
from .data import load_historical_groups, public_bundle, select_roots
from .prompts import PROMPTS
from .schema import (
    Invalid,
    digest,
    require,
    text_hash,
    validate_certificate,
    validate_root,
)

DEFAULT_CONFIG = {
    "schema": "eci-proof-config/v1",
    "version": __version__,
    "threshold": 0.39,
    "seed": 20261027,
    "max_sources": 120,
    "max_proposals_per_source": 2,
    "max_additions_per_source": 1,
    "max_additions_per_document": 5,
    "substituted_endpoint": "cause_only",
    "recursive": False,
    "max_usd": 10.0,
    "max_billed_tokens": 1500000,
    "model": "openai/gpt-6-luna:batch",
    "expected_version": "gpt-6-luna-20260922",
    "reasoning_effort": "high",
    "max_output_tokens": 8192,
    "endpoint": "https://openrouter.ai/api/v1/batches",
    "transport": "batch_inline",
    "batch_size": 8,
    "completion_window": "24h",
    "api_key_env": "OPENROUTER_API_KEY",
    "paid_authorization_required": True,
    "temperature": None,
    "retries": 0,
    "selection": "topic_doc_round_robin_score_hash",
    "dataset_status": "DEV44_consumed_development_not_confirmation",
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_once(path: Path, value) -> None:
    data = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(
            path.read_text() == data, "REFUSE_TO_OVERWRITE_FROZEN_ARTIFACT:" + str(path)
        )
    else:
        with path.open("x", encoding="utf-8") as f:
            f.write(data)


def read_jsonl(path: Path) -> list:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl_once(path: Path, values: list) -> None:
    data = "".join(
        json.dumps(x, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
        for x in values
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(
            path.read_text() == data, "REFUSE_TO_OVERWRITE_FROZEN_ARTIFACT:" + str(path)
        )
    else:
        with path.open("x", encoding="utf-8") as f:
            f.write(data)


def validate_config(config: dict) -> None:
    require(set(config) == set(DEFAULT_CONFIG), "CONFIG_SCHEMA")
    # Scientific rules are constants. No hidden threshold/cap grid is supported.
    for key in (
        "schema",
        "version",
        "threshold",
        "seed",
        "max_proposals_per_source",
        "max_additions_per_source",
        "max_additions_per_document",
        "substituted_endpoint",
        "recursive",
        "selection",
        "paid_authorization_required",
        "retries",
        "dataset_status",
        "transport",
        "batch_size",
        "completion_window",
    ):
        require(config[key] == DEFAULT_CONFIG[key], "FROZEN_RULE_CHANGED:" + key)
    require(
        type(config["max_sources"]) is int and 0 < config["max_sources"] <= 120,
        "SOURCE_LIMIT",
    )
    require(
        type(config["max_billed_tokens"]) is int
        and 0 < config["max_billed_tokens"] <= 1500000,
        "TOKEN_LIMIT",
    )
    require(
        type(config["max_usd"]) in (int, float) and 0 < config["max_usd"] <= 10,
        "DOLLAR_LIMIT",
    )
    require(config["endpoint"] == DEFAULT_CONFIG["endpoint"], "UNAPPROVED_ENDPOINT")
    require(config["temperature"] is None, "UNSUPPORTED_TEMPERATURE")
    require(
        type(config["max_output_tokens"]) is int
        and 0 < config["max_output_tokens"] <= 8192,
        "OUTPUT_LIMIT",
    )
    require(
        config["model"] == DEFAULT_CONFIG["model"]
        and config["expected_version"] == DEFAULT_CONFIG["expected_version"]
        and config["reasoning_effort"] == "high",
        "MODEL_SETTING_CHANGED",
    )


def implementation_hashes() -> dict:
    package = Path(__file__).parent
    paths = sorted(package.glob("*.py"))
    paths += [
        p
        for p in (package.parent / "pyproject.toml", package.parent / "uv.lock")
        if p.exists()
    ]
    return {p.name: text_hash(p.read_text()) for p in paths}


def prepare(
    suite: dict,
    controls: list,
    group_dir: Path,
    output: Path,
    *,
    config: dict | None = None,
    provenance: dict | None = None,
) -> dict:
    config = dict(DEFAULT_CONFIG if config is None else config)
    validate_config(config)
    bundle = public_bundle(suite, controls)
    roots = select_roots(bundle, config["max_sources"], config["seed"])
    groups = load_historical_groups(group_dir, bundle["documents"])
    values = {
        "public.json": bundle,
        "roots.json": roots,
        "historical_groups.json": groups,
        "config.json": config,
        "prompts.json": PROMPTS,
    }
    for name, value in values.items():
        write_once(output / name, value)
    manifest = {
        "schema": "eci-proof-plan/v1",
        "files": {name: digest(value) for name, value in values.items()},
        "implementation_hashes": implementation_hashes(),
        "provenance": provenance or {},
        "source_count": len(roots),
        "candidate_registry_status": "PENDING_COMMON_MODEL_STAGES",
        "candidate_capacity_not_actual_candidates": 2 * len(roots),
        "documents": len(bundle["documents"]),
        "selected_documents": len({r["doc"] for r in roots}),
        "historical_T2_available": sum(
            g["status"] == "AVAILABLE" for g in groups.values()
        ),
        "historical_T2_missing": {
            d: g["reason"] for d, g in groups.items() if g["status"] != "AVAILABLE"
        },
        "uses_gold_for_selection": False,
        "baseline_modified": False,
        "old_88_pristine": False,
    }
    write_once(output / "plan.json", manifest)
    # Capacity slots explicitly have NO alternative or direction until paid stages.
    slots = [
        {
            "root_id": r["id"],
            "slot": i,
            "status": "PENDING_SOURCE_ORIENTATION_AND_PROPOSAL",
            "alternative": None,
        }
        for r in roots
        for i in range(2)
    ]
    write_jsonl_once(output / "candidate-slots.pending.jsonl", slots)
    return manifest


def load_plan(path: Path) -> tuple[dict, dict, list, dict, dict]:
    manifest = read_json(path / "plan.json")
    require(
        manifest["implementation_hashes"] == implementation_hashes(),
        "IMPLEMENTATION_DRIFT_AFTER_FREEZE",
    )
    require(
        set(manifest["files"])
        == {
            "public.json",
            "roots.json",
            "historical_groups.json",
            "config.json",
            "prompts.json",
        },
        "PLAN_FILE_SCHEMA",
    )
    values = {name: read_json(path / name) for name in manifest["files"]}
    require(
        all(digest(v) == manifest["files"][k] for k, v in values.items()),
        "PLAN_INTEGRITY",
    )
    require(values["prompts.json"] == PROMPTS, "PROMPT_CODE_DRIFT")
    validate_config(values["config.json"])
    bundle, roots = values["public.json"], values["roots.json"]
    # Strictly reconstruct the public schema without importing labels.
    require(
        set(bundle) == {"schema", "threshold", "documents", "scores"}
        and bundle["threshold"] == 0.39,
        "PUBLIC_BUNDLE_SCHEMA",
    )
    from .schema import validate_document

    for doc in bundle["documents"].values():
        validate_document(doc)
    for root in roots:
        validate_root(root, bundle)
    return (
        manifest,
        bundle,
        roots,
        values["historical_groups.json"],
        values["config.json"],
    )


def model_document(doc: dict) -> dict:
    return {
        "id": doc["id"],
        "sha256": doc["sha256"],
        "text": doc["text"],
        "offset_unit": doc["offset_unit"],
        "mentions": doc["mentions"],
        "sentences": [
            {
                "id": i,
                "start": doc["token_offsets"][a][0],
                "end": doc["token_offsets"][b - 1][1],
            }
            for i, (a, b) in enumerate(doc["sentences"])
        ],
    }


def request(
    stage: str,
    did: str,
    document: dict,
    items: list,
    manifest: dict,
    config: dict,
    *,
    registry_hash: str | None = None,
) -> dict:
    value = {
        "stage": stage,
        "doc": did,
        "plan_hash": digest(manifest),
        "registry_hash": registry_hash,
        "model": config["model"],
        "expected_version": config["expected_version"],
        "reasoning_effort": config["reasoning_effort"],
        "max_output_tokens": config["max_output_tokens"],
        "messages": [
            {"role": "system", "content": PROMPTS[stage]},
            {
                "role": "user",
                "content": json.dumps(
                    {"document": model_document(document), "items": items},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            },
        ],
    }
    value["id"] = digest(value)
    return value


def extraction_requests(plan: Path) -> list:
    manifest, bundle, roots, _, config = load_plan(plan)
    by_doc = defaultdict(list)
    for root in roots:
        by_doc[root["doc"]].append({"source_id": root["id"], "pair": root["pair"]})
    return [
        request("extract", d, bundle["documents"][d], rows, manifest, config)
        for d, rows in sorted(by_doc.items())
    ]


def collect_items(requests: list, receipts: list, id_field: str) -> tuple[dict, list]:
    expected = {r["id"]: r for r in requests}
    require(
        len({r["request_id"] for r in receipts}) == len(receipts), "DUPLICATE_RECEIPT"
    )
    require({r["request_id"] for r in receipts} == set(expected), "RECEIPT_COVERAGE")
    out, usage = {}, []
    for receipt in receipts:
        req = expected[receipt["request_id"]]
        require(
            receipt["request_hash"] == digest(req) and receipt["model"] == req["model"],
            "RECEIPT_REQUEST_OR_MODEL_MISMATCH",
        )
        require(receipt["mode"] in {"live", "synthetic_fixture"}, "RECEIPT_MODE")
        if receipt["mode"] == "synthetic_fixture":
            require(
                req["doc"].startswith("SYNTHETIC_"),
                "SYNTHETIC_RECEIPTS_ON_REAL_DOCUMENTS",
            )
        if receipt["mode"] == "live":
            require(
                isinstance(receipt.get("served_model"), str)
                and bool(receipt["served_model"]),
                "MISSING_SERVED_MODEL",
            )
            require("execution_error" not in receipt, "FAILED_MODEL_EXECUTION")
        require(
            isinstance(receipt["response"], dict)
            and set(receipt["response"]) == {"items"},
            "RESPONSE_SCHEMA",
        )
        rows = receipt["response"]["items"]
        require(
            isinstance(rows, list)
            and all(isinstance(x, dict) and id_field in x for x in rows),
            "RESPONSE_ITEMS",
        )
        wanted = {
            x[id_field] for x in json.loads(req["messages"][1]["content"])["items"]
        }
        require(
            len(rows) == len(wanted) and {x[id_field] for x in rows} == wanted,
            "RESPONSE_ITEM_COVERAGE",
        )
        for row in rows:
            require(row[id_field] not in out, "DUPLICATE_RESPONSE_ITEM")
            out[row[id_field]] = row
        u = receipt["usage"]
        require(
            set(u) == {"input_tokens", "output_tokens", "cost_usd"}
            and type(u["input_tokens"]) is int
            and type(u["output_tokens"]) is int
            and u["input_tokens"] >= 0
            and u["output_tokens"] >= 0
            and type(u["cost_usd"]) in (int, float)
            and u["cost_usd"] >= 0,
            "USAGE_SCHEMA",
        )
        usage.append(
            {
                "request_id": req["id"],
                "stage": req["stage"],
                "mode": receipt["mode"],
                **(
                    {
                        "batch_id": receipt["batch_id"],
                        "transport": receipt.get("transport"),
                        "cost_kind": receipt.get("cost_kind"),
                        "token_usage_kind": receipt.get("token_usage_kind"),
                    }
                    if "batch_id" in receipt
                    else {}
                ),
                **u,
            }
        )
    return out, usage


def proposal_requests(plan: Path, extraction_receipts: list) -> tuple[list, dict]:
    manifest, bundle, roots, _, config = load_plan(plan)
    extracted, _ = collect_items(
        extraction_requests(plan), extraction_receipts, "source_id"
    )
    by_doc = defaultdict(list)
    invalid_citation_items = 0
    for root in roots:
        item, doc = extracted[root["id"]], bundle["documents"][root["doc"]]
        if item.get("verdict") != "SUPPORTED":
            continue
        try:
            validate_certificate(item["certificate"], root, doc)
        except (Invalid, KeyError, TypeError) as error:
            if "CITATION" in str(error):
                invalid_citation_items += 1
            continue
        cert = item["certificate"]
        by_doc[root["doc"]].append(
            {
                "source_id": root["id"],
                "source_certificate_hash": digest(cert),
                "certificate": cert,
            }
        )
    require(
        invalid_citation_items / max(1, len(roots)) <= 0.05,
        "CITATION_ERROR_RATE_EXCEEDS_5_PERCENT_STOP",
    )
    reqs = [
        request("propose", d, bundle["documents"][d], rows, manifest, config)
        for d, rows in sorted(by_doc.items())
    ]
    return reqs, extracted


def freeze_registry(
    plan: Path, extraction_receipts: list, proposal_receipts: list, output: Path
) -> dict:
    manifest, bundle, roots, _, _ = load_plan(plan)
    reqs, extracted = proposal_requests(plan, extraction_receipts)
    verified, _ = collect_items(reqs, proposal_receipts, "source_id")
    registry = build_registry(bundle, roots, extracted, verified)
    citation_errors = sum(
        "CITATION" in v.get("reason", "") for v in registry["sources"].values()
    )
    require(
        citation_errors / max(1, len(roots)) <= 0.05,
        "CITATION_ERROR_RATE_EXCEEDS_5_PERCENT_STOP",
    )
    validate_registry(bundle, roots, registry)
    write_once(output / "registry.json", registry)
    write_jsonl_once(output / "candidates.frozen.jsonl", registry["proposals"])
    freeze = {
        "schema": "eci-proof-freeze/v1",
        "plan_hash": digest(manifest),
        "registry_hash": digest(registry),
        "extraction_receipts_hash": digest(extraction_receipts),
        "proposal_receipts_hash": digest(proposal_receipts),
        "created_utc": datetime.now(UTC).isoformat(),
        "candidate_count": len(registry["proposals"]),
        "labels_consulted": False,
    }
    if (output / "freeze.json").exists():
        previous = read_json(output / "freeze.json")
        freeze["created_utc"] = previous["created_utc"]
    write_once(output / "freeze.json", freeze)
    return registry


def judge_requests(plan: Path, registry: dict) -> list:
    manifest, bundle, roots, _, config = load_plan(plan)
    validate_registry(bundle, roots, registry)
    requests = []
    for variant in ("A0", "A1", "A2"):
        by_doc = defaultdict(list)
        for proposal in registry["proposals"]:
            cert = registry["sources"][proposal["root_id"]]["certificate"]
            row = {
                "candidate_id": proposal["id"],
                "cause": proposal["cause"],
                "alternative": proposal["alternative"],
                "effect": proposal["effect"],
            }
            if variant == "A1":
                row["source_cause_frame"] = cert["frames"][cert["cause"]]
            if variant == "A2":
                row.update(
                    source_certificate_hash=digest(cert),
                    evidence_hash=digest(cert["evidence"]),
                    certificate=cert,
                )
            by_doc[proposal["doc"]].append(row)
        requests.extend(
            request(
                variant,
                d,
                bundle["documents"][d],
                rows,
                manifest,
                config,
                registry_hash=digest(registry),
            )
            for d, rows in sorted(by_doc.items())
        )
    return requests


def replay(
    plan: Path,
    registry_dir: Path,
    extraction_receipts: list,
    proposal_receipts: list,
    judge_receipts: list,
    output: Path,
) -> dict:
    manifest, bundle, roots, groups, _ = load_plan(plan)
    registry, freeze = (
        read_json(registry_dir / "registry.json"),
        read_json(registry_dir / "freeze.json"),
    )
    require(
        freeze["plan_hash"] == digest(manifest)
        and freeze["registry_hash"] == digest(registry),
        "FREEZE_INTEGRITY",
    )
    require(
        freeze["extraction_receipts_hash"] == digest(extraction_receipts)
        and freeze["proposal_receipts_hash"] == digest(proposal_receipts),
        "COMMON_RECEIPTS_CHANGED",
    )
    validate_registry(bundle, roots, registry)
    _, use0 = collect_items(extraction_requests(plan), extraction_receipts, "source_id")
    req1, _ = proposal_requests(plan, extraction_receipts)
    _, use1 = collect_items(req1, proposal_receipts, "source_id")
    requests = judge_requests(plan, registry)
    raw_by_variant, usages = {}, use0 + use1
    for variant in ("A0", "A1", "A2"):
        reqs = [r for r in requests if r["stage"] == variant]
        wanted = {r["id"] for r in reqs}
        rows = [r for r in judge_receipts if r["request_id"] in wanted]
        raw_by_variant[variant], use = collect_items(reqs, rows, "candidate_id")
        usages.extend(use)
    require(
        {r["request_id"] for r in judge_receipts} == {r["id"] for r in requests},
        "EXTRA_JUDGE_RECEIPTS",
    )
    decisions = [
        decide(v, p, registry, bundle, raw_by_variant.get(v, {}).get(p["id"]), groups)
        for v in VARIANTS
        for p in registry["proposals"]
    ]
    selected = select_additions(decisions, registry, bundle)
    modes = {u["mode"] for u in usages}
    require(len(modes) <= 1, "MIXED_SYNTHETIC_AND_REAL_RECEIPTS")
    all_receipts = extraction_receipts + proposal_receipts + judge_receipts
    served = {r["served_model"] for r in all_receipts if r["mode"] == "live"}
    require(len(served) <= 1, "SERVED_MODEL_DRIFT_BETWEEN_STAGES")
    mode = next(iter(modes), "no_calls_empty_registry")
    if mode == "synthetic_fixture":
        require(
            all(d.startswith("SYNTHETIC_") for d in bundle["documents"]),
            "SYNTHETIC_RECEIPTS_ON_REAL_DOCUMENTS",
        )
    write_jsonl_once(output / "decisions.jsonl", selected)
    write_jsonl_once(output / "raw-judge-receipts.jsonl", judge_receipts)
    write_jsonl_once(output / "usage.jsonl", usages)
    citation_errors = sum(
        "CITATION" in d["reason"] for d in selected if d["variant"] != "A3"
    )
    model_items = sum(d["variant"] != "A3" for d in selected)
    technical_gate = citation_errors / max(1, model_items) <= 0.05
    run = {
        "schema": "eci-proof-run/v1",
        "plan_hash": digest(manifest),
        "registry_hash": digest(registry),
        "decisions_hash": digest(selected),
        "usage_hash": digest(usages),
        "mode": mode,
        "served_models": sorted(served),
        "expected_version_verified": all(
            r.get("version_verified") is True
            for r in all_receipts
            if r["mode"] == "live"
        ),
        "complete": all(d["verdict"] not in {"PENDING"} for d in selected)
        and technical_gate,
        "citation_integrity_gate": {
            "passed": technical_gate,
            "invalid_logical_items": citation_errors,
            "model_items": model_items,
            "maximum_rate": 0.05,
        },
        "development_only": True,
        "baseline_modified": False,
    }
    write_once(output / "run.json", run)
    return run
