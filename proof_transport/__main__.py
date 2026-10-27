"""Run ``uv run --locked python -m proof_transport --help``. Offline by default."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .pipeline import (
    DEFAULT_CONFIG,
    extraction_requests,
    freeze_registry,
    judge_requests,
    load_plan,
    prepare,
    proposal_requests,
    read_json,
    read_jsonl,
    replay,
    write_jsonl_once,
    write_once,
)
from .schema import Invalid, digest, require, text_hash


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser(
        "prepare-local",
        help="Allowlist historical DEV44 exports; NO model calls, no gold scoring",
    )
    prep.add_argument("--repo", type=Path, default=Path.cwd())
    prep.add_argument("--historical-ref", default="orx/r4-a-luna-event-semantics-dev44")
    prep.add_argument("--artifacts", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--config", type=Path)
    audit = sub.add_parser(
        "audit-test-s40", help="Explicit evaluation-only baseline reproduction"
    )
    audit.add_argument("--pairs", type=Path, required=True)
    audit.add_argument("--raw-rows", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    sm = sub.add_parser(
        "smoke", help="Synthetic fake receipts; NO labels, NO provider calls"
    )
    sm.add_argument("--output", type=Path, required=True)
    req = sub.add_parser(
        "requests", help="Build immutable JSONL requests, never send them"
    )
    req.add_argument("--plan", type=Path, required=True)
    req.add_argument(
        "--stage", choices=["extract", "propose", "judge", "diagnostic"], required=True
    )
    req.add_argument("--extraction-receipts", type=Path)
    req.add_argument("--registry", type=Path)
    req.add_argument("--output", type=Path, required=True)
    fr = sub.add_parser(
        "freeze", help="Freeze actual candidates after common source stages"
    )
    rp = sub.add_parser(
        "replay",
        help="Apply all arms and caps to complete recorded responses; no network",
    )
    for cmd in (fr, rp):
        cmd.add_argument("--plan", type=Path, required=True)
        cmd.add_argument("--extraction-receipts", type=Path, required=True)
        cmd.add_argument("--proposal-receipts", type=Path, required=True)
        cmd.add_argument("--output", type=Path, required=True)
    rp.add_argument("--registry-dir", type=Path, required=True)
    rp.add_argument("--judge-receipts", type=Path, required=True)
    pricing = sub.add_parser(
        "pricing",
        help="Select public tariff from an already downloaded catalog; NO model calls",
    )
    pricing.add_argument("--catalog", type=Path, required=True)
    pricing.add_argument("--output", type=Path, required=True)
    live = sub.add_parser(
        "execute",
        help="PAID; refuses without explicit external approval and fresh bound tariffs",
    )
    live.add_argument("--plan", type=Path, required=True)
    live.add_argument("--requests", type=Path, required=True)
    live.add_argument("--pricing", type=Path, required=True)
    live.add_argument("--approval", type=Path, required=True)
    live.add_argument("--ledger", type=Path, required=True)
    live.add_argument("--output", type=Path, required=True)
    ba = sub.add_parser(
        "blind-audit",
        help="Export all proposals without labels, scores, acceptance or arm names",
    )
    ba.add_argument("--plan", type=Path, required=True)
    ba.add_argument("--registry", type=Path, required=True)
    ba.add_argument("--output", type=Path, required=True)
    ba.add_argument(
        "--judge-receipts",
        type=Path,
        help="Include actual transport evidence without showing decisions or labels",
    )
    native = sub.add_parser(
        "native-t2",
        help="Label-free exact historical group maximum diagnostic, not controlled A3",
    )
    native.add_argument("--plan", type=Path, required=True)
    native.add_argument("--output", type=Path, required=True)
    parity = sub.add_parser(
        "validate-t2-parity",
        help="Compare native T2 with the reviewed archived operator, without gold",
    )
    parity.add_argument("--plan", type=Path, required=True)
    parity.add_argument("--archived-script", type=Path, required=True)
    parity.add_argument("--output", type=Path, required=True)
    diagnostic = sub.add_parser(
        "inspect-diagnostics",
        help="Inspect raw witness-dependence obligations, without labels",
    )
    diagnostic.add_argument("--plan", type=Path, required=True)
    diagnostic.add_argument("--registry", type=Path, required=True)
    diagnostic.add_argument("--receipts", type=Path, required=True)
    diagnostic.add_argument("--output", type=Path, required=True)
    evgold = sub.add_parser(
        "export-evaluation-gold", help="Evaluation enclave only; requires sealed run"
    )
    ev = sub.add_parser(
        "evaluate", help="Score sealed outputs; NEVER called by inference"
    )
    for cmd in (evgold, ev):
        cmd.add_argument("--plan", type=Path, required=True)
        cmd.add_argument("--run-dir", type=Path, required=True)
        cmd.add_argument("--output", type=Path, required=True)
    evgold.add_argument("--arrow", type=Path, nargs="+", required=True)
    ev.add_argument("--registry-dir", type=Path, required=True)
    ev.add_argument("--gold", type=Path, required=True)
    ev.add_argument("--human-audit", type=Path)
    ev.add_argument("--audit-id-map", type=Path)
    ev.add_argument("--diagnostic-receipts", type=Path)
    a = p.parse_args(argv)
    if a.command == "prepare-local":
        commit = subprocess.check_output(
            ["git", "-C", str(a.repo), "rev-parse", a.historical_ref], text=True
        ).strip()
        load = lambda path: subprocess.check_output(
            ["git", "-C", str(a.repo), "show", commit + ":" + path], text=True
        )
        suite_text = load("experiments/jev/round4/dev44-suite.json")
        control_text = load("experiments/jev/round4/controls.json")
        result = prepare(
            json.loads(suite_text),
            json.loads(control_text),
            a.artifacts / "jev-v1/round3/runs/T2",
            a.output,
            config=read_json(a.config) if a.config else None,
            provenance={
                "git_commit": commit,
                "historical_ref": a.historical_ref,
                "mixed_suite_sha256_utf8": text_hash(suite_text),
                "mixed_control_sha256_utf8": text_hash(control_text),
                "adapter": "strict allowlist; gold ignored and not exported to inference",
                "source_report": "transport-preuve-locale-conception-2026-10-27.md",
            },
        )
    elif a.command == "audit-test-s40":
        from .evaluation import audit_test_s40

        result = audit_test_s40(a.pairs, a.raw_rows)
        write_once(a.output, result)
    elif a.command == "smoke":
        from .fixtures import smoke

        result = smoke(a.output)
    elif a.command == "requests":
        if a.stage == "extract":
            result = extraction_requests(a.plan)
        elif a.stage == "propose":
            require(a.extraction_receipts is not None, "EXTRACTION_RECEIPTS_REQUIRED")
            result, _ = proposal_requests(a.plan, read_jsonl(a.extraction_receipts))
        else:
            require(a.registry is not None, "FROZEN_REGISTRY_REQUIRED")
            if a.stage == "judge":
                result = judge_requests(a.plan, read_json(a.registry))
            else:
                from .diagnostics import diagnostic_requests

                result = diagnostic_requests(a.plan, read_json(a.registry))
        write_jsonl_once(a.output, result)
        result = {"requests": len(result), "model_calls": 0, "output": str(a.output)}
    elif a.command == "freeze":
        result = freeze_registry(
            a.plan,
            read_jsonl(a.extraction_receipts),
            read_jsonl(a.proposal_receipts),
            a.output,
        )
        result = {
            "candidates": len(result["proposals"]),
            "registry_hash": digest(result),
            "model_calls": 0,
        }
    elif a.command == "replay":
        result = replay(
            a.plan,
            a.registry_dir,
            read_jsonl(a.extraction_receipts),
            read_jsonl(a.proposal_receipts),
            read_jsonl(a.judge_receipts),
            a.output,
        )
    elif a.command == "pricing":
        from .client import tariff_snapshot

        result = tariff_snapshot(a.catalog.read_text(), DEFAULT_CONFIG["model"])
        write_once(a.output, result)
    elif a.command == "execute":
        from .client import execute

        manifest, _, _, _, config = load_plan(a.plan)
        result = execute(
            read_jsonl(a.requests),
            manifest,
            config,
            read_json(a.pricing),
            read_json(a.approval),
            a.ledger,
            a.output,
        )
        result = {"completed_requests": len(result), "output": str(a.output)}
    elif a.command == "blind-audit":
        from .evaluation import blind_audit

        blind_audit(
            a.plan,
            read_json(a.registry),
            a.output,
            read_jsonl(a.judge_receipts) if a.judge_receipts else None,
        )
        result = {"output": str(a.output), "labels_exported": False}
    elif a.command == "native-t2":
        from .historical import native_t2

        _, bundle, _, groups, _ = load_plan(a.plan)
        result = native_t2(bundle, groups)
        write_once(a.output, result)
        result = {"output": str(a.output), "model_calls": 0, "is_controlled_A3": False}
    elif a.command == "validate-t2-parity":
        from .validation import historical_t2_parity

        _, bundle, _, groups, _ = load_plan(a.plan)
        result = historical_t2_parity(bundle, groups, a.archived_script)
        write_once(a.output, result)
    elif a.command == "inspect-diagnostics":
        from .diagnostics import inspect_diagnostics

        result, usage = inspect_diagnostics(
            a.plan, read_json(a.registry), read_jsonl(a.receipts)
        )
        write_once(a.output / "witness-dependence.json", result)
        write_jsonl_once(a.output / "diagnostic-usage.jsonl", usage)
    elif a.command in {"export-evaluation-gold", "evaluate"}:
        run = read_json(a.run_dir / "run.json")
        require(
            run["complete"] is True
            and run["decisions_hash"]
            == digest(read_jsonl(a.run_dir / "decisions.jsonl")),
            "SEALED_RUN_REQUIRED_BEFORE_GOLD",
        )
        if a.command == "export-evaluation-gold":
            from .evaluation import export_gold_from_arrow

            _, bundle, _, _, _ = load_plan(a.plan)
            result = export_gold_from_arrow(a.arrow, set(bundle["documents"]))
            write_once(a.output, result)
        else:
            from .evaluation import evaluate

            human = None
            if a.human_audit:
                require(a.audit_id_map is not None, "AUDIT_ID_MAP_REQUIRED")
                mapping = read_json(a.audit_id_map)
                human = {}
                for r in read_jsonl(a.human_audit):
                    require(
                        set(r)
                        == {
                            "audit_id",
                            "source_valid",
                            "binding_valid",
                            "scope_valid",
                            "proof_valid",
                        },
                        "HUMAN_AUDIT_SCHEMA",
                    )
                    require(
                        r["audit_id"] in mapping
                        and mapping[r["audit_id"]] not in human,
                        "HUMAN_AUDIT_ID",
                    )
                    require(
                        all(
                            r[k] in {"YES", "NO", "UNKNOWN"}
                            for k in r
                            if k != "audit_id"
                        ),
                        "HUMAN_AUDIT_VALUE",
                    )
                    human[mapping[r["audit_id"]]] = r
            result = evaluate(
                a.plan,
                a.registry_dir,
                a.run_dir,
                read_json(a.gold),
                a.output,
                human_audit=human,
                diagnostic_receipts=read_jsonl(a.diagnostic_receipts)
                if a.diagnostic_receipts
                else None,
            )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Invalid, KeyError, TypeError, json.JSONDecodeError) as error:
        print(
            json.dumps(
                {
                    "status": "STOPPED",
                    "reason": str(error),
                    "error_type": type(error).__name__,
                }
            ),
            file=sys.stderr,
        )
        raise SystemExit(2)
