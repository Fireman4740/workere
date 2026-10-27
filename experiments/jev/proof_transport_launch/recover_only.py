"""GET-only recovery of already paid/submitted batch jobs.

This path never calls execute/submit, never generates candidates, and never opens
gold. A blocked scientific job remains blocked. Only other acknowledged IDs are
collected to preserve their results and settle the shared billing ledger.
"""

from __future__ import annotations

import fcntl
import time
from pathlib import Path

from proof_transport import client
from proof_transport.batch import Store, _http
from proof_transport.pipeline import write_once
from proof_transport.schema import Invalid, digest, require

from .controller import budget_status


def get_only(config, method, url, payload=None):
    require(
        method == "GET" and payload is None, "RECOVERY_FORBIDS_POST_OR_NEW_GENERATION"
    )
    return _http(config, method, url)


def pending_jobs(ledger, manifest, approval):
    store = Store(ledger, digest(manifest), approval)
    try:
        jobs = store.db.execute(
            "SELECT id,batch_id,status FROM proof_batch_job WHERE scope=? ORDER BY id",
            (digest(manifest),),
        ).fetchall()
        require(bool(jobs), "NO_DURABLE_BATCH_JOBS_TO_RECOVER")
        pending = [
            (jid, bid, [r for r, _ in store.items(jid)])
            for jid, bid, status in jobs
            if status == "ACKNOWLEDGED"
        ]
        return jobs, pending
    finally:
        store.close()


def recover(
    context, *, collect_fn=client.collect, sleep_fn=time.sleep, clock=time.monotonic
):
    recipe, _, manifest, _, _, _, config, pricing, approval, state = context
    require(recipe.get("mode") == "collect_existing_only", "RECOVERY_MODE_REQUIRED")
    ledger = Path(approval["ledger_path"])
    require(ledger.exists(), "RECOVERY_MUST_NOT_CREATE_FRESH_LEDGER")
    lock = (state / "controller.lock").open("a")
    started = clock()
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        context.emit(
            "GET_ONLY_RECOVERY_STARTED",
            new_model_posts=0,
            read_gold=False,
            budget=budget_status(ledger, digest(manifest)),
        )
        while True:
            require(
                clock() - started <= recipe["max_phase_wait_seconds"],
                "GET_ONLY_RECOVERY_TIMEOUT",
            )
            jobs, pending = pending_jobs(ledger, manifest, approval)
            if not pending:
                uncertain = [
                    bid or jid for jid, bid, status in jobs if status == "SUBMITTING"
                ]
                require(
                    not uncertain, "UNCERTAIN_SUBMISSION_LEFT_FOR_OPERATOR_NO_RESUBMIT"
                )
                result = {
                    "status": "RECEIPTS_RECOVERED_SCIENTIFIC_PILOT_STILL_FAILED",
                    "new_model_posts": 0,
                    "read_gold": False,
                    "scientific_comparison_available": False,
                    "jobs": [
                        {"job_hash": jid, "batch_id": bid, "local_status": status}
                        for jid, bid, status in jobs
                    ],
                    "budget": budget_status(ledger, digest(manifest)),
                }
                write_once(state / "recovery" / "recovery-summary.json", result)
                context.emit("GET_ONLY_RECOVERY_COMPLETE", **result)
                return 0
            for jid, bid, requests in pending:
                require(bid is not None, "RECOVERY_REQUIRES_ACKNOWLEDGED_ID")
                output = state / "recovery" / jid / "receipts.jsonl"
                try:
                    result = collect_fn(
                        requests,
                        manifest,
                        config,
                        pricing,
                        approval,
                        ledger,
                        output,
                        http=lambda method, url, payload=None: get_only(
                            config, method, url, payload
                        ),
                    )
                    context.emit(
                        "GET_ONLY_COLLECTION", batch_id=bid, new_model_posts=0, **result
                    )
                except Invalid as error:
                    # The cost is already settled before this fail-closed signal.
                    # Preserve the scientific block, but still recover OTHER jobs.
                    if (
                        str(error)
                        != "FAILED_BATCH_RESULTS_CHARGED_NO_SCIENTIFIC_OUTPUT"
                    ):
                        raise
                    context.emit(
                        "GET_ONLY_JOB_CHARGED_BUT_SCIENTIFICALLY_BLOCKED",
                        batch_id=bid,
                        reason=str(error),
                        new_model_posts=0,
                        budget=budget_status(ledger, digest(manifest)),
                    )
            # Never run source verification/proposal/judging after recovery.
            if pending_jobs(ledger, manifest, approval)[1]:
                sleep_fn(recipe["poll_interval_seconds"])
    finally:
        lock.close()
