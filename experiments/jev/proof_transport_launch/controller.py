"""Supervised SSH controller for the explicitly authorized frozen batch pilot.

This orchestration layer is outside the hashed inference package. It changes no
model prompt, candidate policy, rule, score or threshold, and never opens gold.
Run through the experiment's fixed launcher; --check is a lightweight no-network
preflight for the committed recipe, approval and data hashes.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sqlite3
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from proof_transport import client
from proof_transport.diagnostics import diagnostic_requests, inspect_diagnostics
from proof_transport.evaluation import blind_audit
from proof_transport.pipeline import (
    extraction_requests,
    freeze_registry,
    judge_requests,
    load_plan,
    proposal_requests,
    read_json,
    read_jsonl,
    replay,
    write_jsonl_once,
    write_once,
)
from proof_transport.schema import Invalid, digest, require

RECIPE = Path("experiments/jev/proof_transport_launch/recipe.json")


def load_recipe(path=RECIPE):
    recipe = read_json(path)
    plan = Path(recipe["plan_directory"])
    manifest, bundle, roots, groups, config = load_plan(plan)
    require(digest(manifest) == recipe["expected_plan_hash"], "LAUNCH_PLAN_DRIFT")
    pricing = read_json(Path(recipe["pricing_file"]))
    approval = read_json(Path(recipe["approval_file"]))
    client.validate_approval(approval, manifest, config, pricing, collection_only=True)
    require(
        approval["model"] == "openai/gpt-6-luna:batch"
        and approval["max_usd"] == recipe["currency_cap_usd"] == 10
        and approval["max_billed_tokens"] == recipe["billed_token_cap"] == 1500000,
        "LAUNCH_AUTHORIZATION_SCOPE",
    )
    state = Path(recipe["state_directory"])
    require(
        state.is_absolute()
        and Path(approval["ledger_path"]) == state / "budget.sqlite",
        "LAUNCH_SHARED_LEDGER_PATH",
    )
    return (
        recipe,
        plan,
        manifest,
        bundle,
        roots,
        groups,
        config,
        pricing,
        approval,
        state,
    )


def budget_status(ledger: Path, scope: str):
    if not ledger.exists():
        return {
            "states": {},
            "accounted_or_reserved_usd": 0,
            "accounted_or_reserved_tokens": 0,
        }
    db = sqlite3.connect("file:" + str(ledger) + "?mode=ro", uri=True)
    try:
        rows = db.execute(
            "SELECT status,COUNT(*),SUM(usd),SUM(tokens) FROM request WHERE scope=? GROUP BY status",
            (scope,),
        ).fetchall()
    finally:
        db.close()
    return {
        "states": {
            s: {"logical_requests": n, "usd": u, "tokens": t} for s, n, u, t in rows
        },
        "accounted_or_reserved_usd": sum(r[2] for r in rows),
        "accounted_or_reserved_tokens": sum(r[3] for r in rows),
    }


def drive_stage(
    stage,
    requests,
    context,
    *,
    collect_fn=client.collect,
    submit_fn=client.execute,
    sleep_fn=time.sleep,
    clock=time.monotonic,
):
    recipe, _, manifest, _, _, _, config, pricing, approval, state = context
    output = state / (stage + ".receipts.jsonl")
    ledger = Path(approval["ledger_path"])
    write_jsonl_once(state / (stage + ".requests.jsonl"), requests)
    started = clock()
    emit = context.emit
    emit("PHASE_STARTED", phase=stage, logical_requests=len(requests))
    while True:
        require(
            clock() - started <= recipe["max_phase_wait_seconds"],
            "BATCH_PHASE_TIMEOUT_RECEIPTS_RETAINED",
        )
        status = collect_fn(
            requests, manifest, config, pricing, approval, ledger, output
        )
        emit(
            "COLLECTION",
            phase=stage,
            **status,
            budget=budget_status(ledger, digest(manifest)),
        )
        if status["final_receipts_written"]:
            records = read_jsonl(output)
            require(len(records) == len(requests), "COLLECTED_PHASE_COVERAGE")
            emit(
                "PHASE_COMPLETE",
                phase=stage,
                logical_requests=len(records),
                budget=budget_status(ledger, digest(manifest)),
            )
            return records
        if status["unassigned_requests"]:
            submitted = submit_fn(
                requests, manifest, config, pricing, approval, ledger, output
            )
            emit(
                "SUBMISSION",
                phase=stage,
                **submitted,
                budget=budget_status(ledger, digest(manifest)),
            )
            if submitted["new_batch_posts"] == 0 and not status["pending_batch_ids"]:
                raise Invalid("BUDGET_LIMIT_INCOMPLETE_PHASE_NO_NEGATIVE_DEFAULTS")
        sleep_fn(recipe["poll_interval_seconds"])


class Context(tuple):
    def emit(self, event, **fields):
        row = {
            "utc": datetime.now(UTC).isoformat(),
            "event": event,
            "run_id": os.environ.get("ORX_RUN_ID"),
            **fields,
        }
        print(
            "PROOF_TRANSPORT " + json.dumps(row, ensure_ascii=False, sort_keys=True),
            flush=True,
        )
        with (self[-1] / "controller-events.jsonl").open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="No provider calls, no ledger or approval creation",
    )
    args = parser.parse_args(argv)
    values = load_recipe()
    recipe, plan, manifest, bundle, roots, groups, config, pricing, approval, state = (
        values
    )
    # Credential alias only; never log either value or alter any scientific setting.
    if not os.environ.get(config["api_key_env"]) and os.environ.get("OPENROUTER"):
        os.environ[config["api_key_env"]] = os.environ["OPENROUTER"]
    check = {
        "model": config["model"],
        "plan_hash": digest(manifest),
        "sources": len(roots),
        "documents": len(bundle["documents"]),
        "T2_available": sum(g["status"] == "AVAILABLE" for g in groups.values()),
        "max_usd": approval["max_usd"],
        "max_billed_tokens": approval["max_billed_tokens"],
        "credential_present": bool(os.environ.get(config["api_key_env"])),
        "ledger_path": approval["ledger_path"],
        "read_gold": False,
    }
    if args.check:
        client.validate_approval(
            approval,
            manifest,
            config,
            pricing,
            collection_only=recipe.get("mode") == "collect_existing_only",
        )
        print(
            json.dumps(
                {"status": "PREFLIGHT_ONLY_NO_PROVIDER_CALLS", **check}, sort_keys=True
            )
        )
        return 0
    require(
        check["credential_present"], "MISSING_PROVIDER_CREDENTIAL_BEFORE_SUBMISSION"
    )
    if recipe.get("mode") == "collect_existing_only":
        from .recover_only import recover

        require(state.is_dir(), "RECOVERY_REQUIRES_EXISTING_STATE")
        context = Context(values)
        try:
            return recover(context)
        except Exception as error:
            context.emit(
                "GET_ONLY_RECOVERY_STOPPED",
                error_type=type(error).__name__,
                reason=str(error),
                new_model_posts=0,
                budget=budget_status(Path(approval["ledger_path"]), digest(manifest)),
            )
            raise
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    context = Context(values)
    lock = (state / "controller.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_once(
            state / "deployment" / (os.environ.get("ORX_RUN_ID", "manual") + ".json"),
            {**check, "commit": os.environ.get("ORX_COMMIT"), "recipe": recipe},
        )
        write_once(state / "approval.json", approval)
        write_once(state / "pricing.json", pricing)
        context.emit("AUTHORIZED_START", **check)
        r0 = drive_stage("extract", extraction_requests(plan), context)
        req1, _ = proposal_requests(plan, r0)
        r1 = drive_stage("propose", req1, context)
        registry = freeze_registry(plan, r0, r1, state / "registry")
        context.emit(
            "CANDIDATES_FROZEN",
            candidates=len(registry["proposals"]),
            sources_supported=sum(
                v["status"] == "SUPPORTED" for v in registry["sources"].values()
            ),
            registry_hash=digest(registry),
            labels_consulted=False,
        )
        judges = drive_stage("judge", judge_requests(plan, registry), context)
        run = replay(plan, state / "registry", r0, r1, judges, state / "run")
        require(run["complete"], "TECHNICAL_CITATION_GATE_FAILED_NO_GOLD_SCORING")
        diagnostic = drive_stage(
            "diagnostic", diagnostic_requests(plan, registry), context
        )
        diagnostic_summary, diagnostic_usage = inspect_diagnostics(
            plan, registry, diagnostic
        )
        write_once(state / "diagnostic" / "witness-dependence.json", diagnostic_summary)
        write_jsonl_once(state / "diagnostic" / "usage.jsonl", diagnostic_usage)
        blind_audit(plan, registry, state / "blind-audit", judges)
        decisions = read_jsonl(state / "run" / "decisions.jsonl")
        summary = {
            "status": "SEALED_INFERENCE_READY_FOR_BLIND_AUDIT_AND_EVALUATION",
            "plan_hash": digest(manifest),
            "registry_hash": digest(registry),
            "candidates": len(registry["proposals"]),
            "variant_additions": dict(
                Counter(d["variant"] for d in decisions if d["selected"])
            ),
            "variant_abstentions": dict(
                Counter(d["variant"] for d in decisions if d["verdict"] == "ABSTAIN")
            ),
            "state_directory": str(state),
            "read_gold": False,
            "budget": budget_status(Path(approval["ledger_path"]), digest(manifest)),
            "diagnostic": diagnostic_summary,
            "not_confirmed_gain": True,
        }
        write_once(state / "inference-summary.json", summary)
        context.emit("INFERENCE_COMPLETE", **summary)
        return 0
    except Exception as error:
        context.emit(
            "STOPPED",
            error_type=type(error).__name__,
            reason=str(error),
            budget=budget_status(Path(approval["ledger_path"]), digest(manifest)),
            state_directory=str(state),
            read_gold=False,
        )
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
