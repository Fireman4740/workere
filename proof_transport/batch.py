"""Inline OpenRouter Batch API, isolated from the baseline.

Only submit_batches can POST. Collection GETs only durable acknowledged IDs;
ambiguous submissions never get a second POST. Dollar AND token reservations
cover the whole job before submission and remain held if usage is unresolved.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .client import Budget, upper_bound, validate_approval
from .pipeline import write_jsonl_once, write_once
from .schema import Invalid, digest, require

ENDPOINT = "https://openrouter.ai/api/v1/batches"
TERMINAL = {"completed", "failed", "expired", "cancelled"}


def _validate(
    requests, manifest, config, pricing, approval, ledger_path, *, collect=False
):
    validate_approval(approval, manifest, config, pricing, collection_only=collect)
    require(
        config["endpoint"] == ENDPOINT
        and config["model"].endswith(":batch")
        and config["transport"] == "batch_inline"
        and pricing["transport"] == "batch_catalog",
        "BATCH_TRANSPORT_REQUIRED",
    )
    require(
        Path(approval["ledger_path"]).is_absolute()
        and Path(ledger_path).resolve() == Path(approval["ledger_path"]).resolve(),
        "APPROVED_LEDGER_PATH_REQUIRED",
    )
    require(
        type(config["batch_size"]) is int
        and config["batch_size"] == 8
        and config["completion_window"] == "24h",
        "BATCH_SETTINGS",
    )
    require(len({r["id"] for r in requests}) == len(requests), "DUPLICATE_REQUEST")
    for r in requests:
        require(
            r["id"] == digest({k: v for k, v in r.items() if k != "id"})
            and r["plan_hash"] == digest(manifest)
            and r["model"] == config["model"]
            and r["expected_version"] == config["expected_version"]
            and r["reasoning_effort"] == config["reasoning_effort"]
            and r["max_output_tokens"] == config["max_output_tokens"],
            "REQUEST_INTEGRITY",
        )


def _http(config, method, url, payload=None):
    require(url == ENDPOINT or url.startswith(ENDPOINT + "/"), "UNAPPROVED_BATCH_URL")
    key = os.environ.get(config["api_key_env"])
    require(bool(key), "MISSING_API_KEY")
    req = urllib.request.Request(
        url,
        method=method,
        data=json.dumps(payload, ensure_ascii=False).encode()
        if payload is not None
        else None,
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as response:
        return json.load(response)


def payload_for(requests, config):
    """Match the proven native transport: routing header first, requests LAST."""
    return {
        "endpoint": "/v1/chat/completions",
        "model": config["model"].removesuffix(":batch"),
        "completion_window": config["completion_window"],
        "requests": [
            {
                "custom_id": r["id"],
                "body": {
                    "messages": r["messages"],
                    "max_tokens": r["max_output_tokens"],
                    "reasoning": {"effort": r["reasoning_effort"]},
                    "response_format": {"type": "json_object"},
                    "provider": {"allow_fallbacks": False, "require_parameters": True},
                },
            }
            for r in requests
        ],
    }


class Store:
    def __init__(self, path, scope, approval):
        self.budget = Budget(
            Path(path), scope, approval["max_usd"], approval["max_billed_tokens"]
        )
        self.db = self.budget.db
        self.scope = scope
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS proof_batch_job (id TEXT PRIMARY KEY, scope TEXT, stage TEXT, status TEXT, batch_id TEXT, payload TEXT, pricing TEXT, raw TEXT, error TEXT)"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS proof_batch_request (request_id TEXT PRIMARY KEY, job_id TEXT, request TEXT, bound TEXT)"
        )
        self.db.commit()

    def close(self):
        self.budget.close()

    def assigned(self, rid):
        r = self.db.execute(
            "SELECT job_id FROM proof_batch_request WHERE request_id=?", (rid,)
        ).fetchone()
        return r[0] if r else None

    def job(self, jid):
        r = self.db.execute(
            "SELECT id,stage,status,batch_id,payload,pricing,raw,error FROM proof_batch_job WHERE id=? AND scope=?",
            (jid, self.scope),
        ).fetchone()
        require(r is not None, "UNKNOWN_BATCH_JOB")
        keys = (
            "id",
            "stage",
            "status",
            "batch_id",
            "payload",
            "pricing",
            "raw",
            "error",
        )
        x = dict(zip(keys, r))
        for k in ("payload", "pricing", "raw"):
            if x[k] is not None:
                x[k] = json.loads(x[k])
        return x

    def items(self, jid):
        rows = self.db.execute(
            "SELECT request,bound FROM proof_batch_request WHERE job_id=? ORDER BY request_id",
            (jid,),
        ).fetchall()
        return [(json.loads(r), json.loads(b)) for r, b in rows]

    def claim(self, requests, config, pricing):
        body = payload_for(requests, config)
        jid = digest({"scope": self.scope, "payload": body})
        bounds = [upper_bound(r, pricing) for r in requests]
        require(
            all(
                b["billed_token_upper_bound"] <= pricing["context_length"]
                for b in bounds
            ),
            "CONTEXT_LIMIT",
        )
        self.db.execute("BEGIN IMMEDIATE")
        try:
            require(
                not self.db.execute(
                    "SELECT 1 FROM proof_batch_job WHERE scope=? AND status IN ('SUBMITTING','BLOCKED','VIOLATION')",
                    (self.scope,),
                ).fetchone(),
                "UNRESOLVED_BATCH_SUBMISSION_OR_FAILURE",
            )
            require(
                not self.db.execute(
                    "SELECT 1 FROM request WHERE scope=? AND status='VIOLATION'",
                    (self.scope,),
                ).fetchone(),
                "UNRESOLVED_PROVIDER_BUDGET_VIOLATION",
            )
            require(
                all(
                    not self.db.execute(
                        "SELECT 1 FROM request WHERE id=?", (r["id"],)
                    ).fetchone()
                    for r in requests
                ),
                "REQUEST_ALREADY_RESERVED_OR_ASSIGNED",
            )
            spent, tokens = self.db.execute(
                "SELECT COALESCE(SUM(usd),0),COALESCE(SUM(tokens),0) FROM request WHERE scope=?",
                (self.scope,),
            ).fetchone()
            require(
                spent + sum(b["usd_upper_bound"] for b in bounds) <= self.budget.max_usd
                and tokens + sum(b["billed_token_upper_bound"] for b in bounds)
                <= self.budget.max_tokens,
                "BUDGET_EXHAUSTED",
            )
            self.db.execute(
                "INSERT INTO proof_batch_job VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    jid,
                    self.scope,
                    requests[0]["stage"],
                    "SUBMITTING",
                    None,
                    json.dumps(body),
                    json.dumps(pricing),
                    None,
                    None,
                ),
            )
            for r, b in zip(requests, bounds):
                self.db.execute(
                    "INSERT INTO request VALUES (?,?,?,?,?,NULL)",
                    (
                        r["id"],
                        self.scope,
                        b["usd_upper_bound"],
                        b["billed_token_upper_bound"],
                        "RESERVED",
                    ),
                )
                self.db.execute(
                    "INSERT INTO proof_batch_request VALUES (?,?,?,?)",
                    (r["id"], jid, json.dumps(r), json.dumps(b)),
                )
            self.db.commit()
            return jid, body
        except Exception:
            self.db.rollback()
            raise

    def acknowledge(self, jid, raw):
        bid = raw.get("id")
        require(
            isinstance(bid, str) and re.fullmatch(r"[A-Za-z0-9_-]+", bid),
            "BATCH_ID_MISSING_OR_UNSAFE_UNCERTAIN_SUBMISSION",
        )
        self.db.execute(
            "UPDATE proof_batch_job SET status='ACKNOWLEDGED',batch_id=?,raw=? WHERE id=? AND scope=?",
            (bid, json.dumps(raw), jid, self.scope),
        )
        self.db.commit()

    def observed(self, jid, raw):
        require(raw.get("id") == self.job(jid)["batch_id"], "BATCH_ID_CHANGED")
        self.db.execute(
            "UPDATE proof_batch_job SET raw=? WHERE id=?", (json.dumps(raw), jid)
        )
        self.db.commit()

    def cached(self, jid):
        rows = self.db.execute(
            "SELECT q.status,q.receipt FROM request q JOIN proof_batch_request b ON b.request_id=q.id WHERE b.job_id=? ORDER BY q.id",
            (jid,),
        ).fetchall()
        require(all(s == "DONE" and r for s, r in rows), "BATCH_NOT_SETTLED")
        return [json.loads(r) for _, r in rows]

    def block(self, jid, error, raw=None):
        """Retain upper bounds; record any known higher aggregate debit before stop."""
        self.db.execute("BEGIN IMMEDIATE")
        try:
            job = self.job(jid)
            if job["status"] == "DONE":
                self.db.rollback()
                return
            ids = [r["id"] for r, _ in self.items(jid)]
            usd = sum(
                self.db.execute("SELECT usd FROM request WHERE id=?", (i,)).fetchone()[
                    0
                ]
                for i in ids
            )
            try:
                cost = _cost((raw or {}).get("usage", {}).get("cost"))
            except (Invalid, InvalidOperation, TypeError, AttributeError):
                cost = None
            if cost is not None and float(cost) > usd:
                self.db.execute(
                    "UPDATE request SET usd=usd+? WHERE id=?",
                    (float(cost) - usd, ids[0]),
                )
            try:
                token_counts = _tokens((raw or {}).get("usage", {}))
            except (Invalid, TypeError, AttributeError):
                token_counts = None
            if token_counts is not None:
                held = sum(
                    self.db.execute(
                        "SELECT tokens FROM request WHERE id=?", (i,)
                    ).fetchone()[0]
                    for i in ids
                )
                if sum(token_counts) > held:
                    self.db.execute(
                        "UPDATE request SET tokens=tokens+? WHERE id=?",
                        (sum(token_counts) - held, ids[0]),
                    )
            self.db.execute(
                "UPDATE proof_batch_job SET status='BLOCKED',error=?,raw=? WHERE id=?",
                (
                    str(error),
                    json.dumps(raw) if raw is not None else json.dumps(job["raw"]),
                    jid,
                ),
            )
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise

    def settle(self, jid, receipts, raw, *, failed=False):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            if self.job(jid)["status"] == "DONE":
                self.db.rollback()
                return self.cached(jid)
            items = self.items(jid)
            require(
                {r["request_id"] for r in receipts} == {r["id"] for r, _ in items},
                "BATCH_RECEIPT_COVERAGE",
            )
            usd = sum(r["usage"]["cost_usd"] for r in receipts)
            tokens = sum(
                r["usage"]["input_tokens"] + r["usage"]["output_tokens"]
                for r in receipts
            )
            violation = usd > sum(
                b["usd_upper_bound"] for _, b in items
            ) + 1e-12 or tokens > sum(b["billed_token_upper_bound"] for _, b in items)
            state = "VIOLATION" if violation else "BLOCKED" if failed else "DONE"
            for r in receipts:
                u = r["usage"]
                self.db.execute(
                    "UPDATE request SET usd=?,tokens=?,status=?,receipt=? WHERE id=?",
                    (
                        u["cost_usd"],
                        u["input_tokens"] + u["output_tokens"],
                        "VIOLATION" if violation else "DONE",
                        json.dumps(r),
                        r["request_id"],
                    ),
                )
            self.db.execute(
                "UPDATE proof_batch_job SET status=?,raw=?,error=? WHERE id=?",
                (
                    state,
                    json.dumps(raw),
                    "BATCH_RESULT_FAILED" if failed else None,
                    jid,
                ),
            )
            self.db.commit()
            require(not violation, "PROVIDER_EXCEEDED_BATCH_RESERVATION_STOP")
            return receipts
        except Exception:
            if self.db.in_transaction:
                self.db.rollback()
            raise


def _tokens(usage):
    if not isinstance(usage, dict):
        return None
    a = usage.get("prompt_tokens", usage.get("input_tokens"))
    b = usage.get("completion_tokens", usage.get("output_tokens"))
    if a is None or b is None:
        return None
    require(
        type(a) is int and type(b) is int and a >= 0 and b >= 0,
        "INVALID_BATCH_TOKEN_USAGE",
    )
    if "total_tokens" in usage:
        total = usage["total_tokens"]
        require(type(total) is int and total >= a + b, "INVALID_BATCH_TOTAL_USAGE")
        b = total - a
    return a, b


def _cost(value):
    if value is None:
        return None
    require(type(value) in (int, float, str), "INVALID_BATCH_COST")
    x = Decimal(str(value))
    require(x.is_finite() and x >= 0, "INVALID_BATCH_COST")
    return x


def _allocate_tokens(total, weights):
    """Integer allocation preserving the reported aggregate, never an actual per-item claim."""
    if not weights:
        return []
    den = sum(weights)
    out = [total * w // den for w in weights]
    for i in range(total - sum(out)):
        out[i % len(out)] += 1
    return out


def result_receipts(job, items, raw):
    require(raw.get("id") == job["batch_id"], "BATCH_ID_CHANGED")
    rows = raw.get("results")
    require(
        isinstance(rows, list)
        and all(
            isinstance(r, dict) and isinstance(r.get("custom_id"), str) for r in rows
        ),
        "BATCH_RESULTS_SCHEMA",
    )
    by_id = {r["custom_id"]: r for r in rows}
    require(
        len(by_id) == len(rows) and set(by_id) == {r["id"] for r, _ in items},
        "BATCH_RESULT_IDS",
    )
    bodies = [
        (by_id[r["id"]].get("response") or {}).get("body") or {} for r, _ in items
    ]
    require(all(isinstance(b, dict) for b in bodies), "BATCH_RESULT_BODY")
    item_tokens = [_tokens(b.get("usage", {})) for b in bodies]
    aggregate = _tokens(raw.get("usage", {}))
    token_kind = "provider_per_request"
    if any(t is None for t in item_tokens):
        require(aggregate is not None, "MISSING_BATCH_USAGE_RESERVATION_RETAINED")
        # Use known item counts and allocate only the residual to unknown items.
        missing = [i for i, t in enumerate(item_tokens) if t is None]
        residual = [
            aggregate[k] - sum(t[k] for t in item_tokens if t is not None)
            for k in (0, 1)
        ]
        require(min(residual) >= 0, "BATCH_AGGREGATE_USAGE_CONFLICT")
        aa = _allocate_tokens(
            residual[0], [items[i][1]["input_token_upper_bound"] for i in missing]
        )
        bb = _allocate_tokens(
            residual[1], [items[i][1]["output_token_upper_bound"] for i in missing]
        )
        for i, a, b in zip(missing, aa, bb):
            item_tokens[i] = (a, b)
        token_kind = (
            "reported_batch_total_with_residual_allocation_not_per_request_measurement"
        )
    if aggregate is not None:
        require(
            tuple(sum(t[k] for t in item_tokens) for k in (0, 1)) == aggregate,
            "BATCH_AGGREGATE_USAGE_CONFLICT",
        )
    prices = job["pricing"]
    estimated = [
        Decimal(a) * Decimal(prices["conservative_prompt_usd_per_token"])
        + Decimal(b) * Decimal(prices["conservative_completion_usd_per_token"])
        for a, b in item_tokens
    ]
    item_costs = [_cost(b.get("usage", {}).get("cost")) for b in bodies]
    total_cost = _cost(raw.get("usage", {}).get("cost"))
    if total_cost is not None:
        if all(x is not None for x in item_costs) and abs(
            sum(item_costs) - total_cost
        ) <= Decimal("1e-12"):
            costs = item_costs
            cost_kind = "provider_per_request_and_batch_total"
        else:
            den = sum(estimated)
            costs = [
                total_cost * x / den if den else total_cost / len(items)
                for x in estimated
            ]
            costs[-1] += total_cost - sum(costs)
            cost_kind = "tariff_weighted_share_of_reported_batch_total_not_per_request_measurement"
    elif all(x is not None for x in item_costs):
        costs = item_costs
        cost_kind = "provider_per_request"
    else:
        costs = [c if c is not None else e for c, e in zip(item_costs, estimated)]
        cost_kind = "conservative_batch_tariff_where_provider_cost_missing"
    receipts = []
    failed = raw.get("status") != "completed"
    for (req, _), body, tokens, cost in zip(items, bodies, item_tokens, costs):
        response = by_id[req["id"]].get("response") or {}
        record = {
            "request_id": req["id"],
            "request_hash": digest(req),
            "model": req["model"],
            "mode": "live",
            "served_model": body.get("model"),
            "expected_version": req["expected_version"],
            "version_verified": req["expected_version"] in str(body.get("model", "")),
            "batch_id": job["batch_id"],
            "batch_job_hash": job["id"],
            "transport": "batch_inline",
            "response": {},
            "usage": {
                "input_tokens": tokens[0],
                "output_tokens": tokens[1],
                "cost_usd": float(cost),
            },
            "token_usage_kind": token_kind,
            "cost_kind": cost_kind,
            "provider_completion_tokens": body.get("usage", {}).get(
                "completion_tokens"
            ),
            "provider_total_tokens": body.get("usage", {}).get("total_tokens"),
            "provider_reported_batch_cost_usd": raw.get("usage", {}).get("cost"),
            "provider_reported_item_cost_usd": body.get("usage", {}).get("cost"),
            "pricing_hash": digest(prices),
        }
        try:
            require(
                raw.get("status") == "completed"
                and not by_id[req["id"]].get("error")
                and response.get("status_code") == 200,
                "FAILED_BATCH_ITEM_OR_JOB",
            )
            require(
                body.get("model")
                in {
                    req["model"],
                    req["model"].removesuffix(":batch"),
                    req["expected_version"],
                    "openai/" + req["expected_version"],
                },
                "UNEXPECTED_SERVED_MODEL",
            )
            choice = body["choices"][0]
            require(choice["finish_reason"] == "stop", "TRUNCATED_OR_ERROR_RESPONSE")
            record["response"] = json.loads(choice["message"]["content"])
        except (
            Invalid,
            KeyError,
            TypeError,
            IndexError,
            json.JSONDecodeError,
        ) as error:
            record["execution_error"] = str(error)
            failed = True
        receipts.append(record)
    return receipts, failed


def _summary(output, value):
    # Immutable progress snapshots; final receipts are written only after full coverage.
    path = Path(output).parent / (
        Path(output).stem + ".batch-status-" + digest(value)[:16] + ".json"
    )
    write_once(path, value)
    return {**value, "summary_path": str(path)}


def submit_batches(
    requests, manifest, config, pricing, approval, ledger_path, output, *, http=None
):
    _validate(requests, manifest, config, pricing, approval, ledger_path)
    transport = http or (
        lambda method, url, payload=None: _http(config, method, url, payload)
    )
    store = Store(ledger_path, digest(manifest), approval)
    new_jobs = []
    job_ids = set()
    status = "SUBMITTED_OR_ALREADY_ACKNOWLEDGED"
    try:
        # Stage-specific jobs keep arm accounting exact even with aggregate-only costs.
        pending = defaultdict(list)
        for r in requests:
            assigned = store.assigned(r["id"])
            if assigned:
                old = store.job(assigned)
                require(
                    old["batch_id"] is not None
                    and old["status"] not in {"SUBMITTING", "BLOCKED", "VIOLATION"},
                    "UNCERTAIN_OR_FAILED_BATCH_NO_DUPLICATE_POST",
                )
                job_ids.add(assigned)
            else:
                pending[r["stage"]].append(r)
        if pending:
            require(
                http is not None or bool(os.environ.get(config["api_key_env"])),
                "MISSING_API_KEY",
            )
        for stage, rows in sorted(pending.items()):
            rows.sort(key=lambda r: r["id"])
            for i in range(0, len(rows), config["batch_size"]):
                chunk = rows[i : i + config["batch_size"]]
                try:
                    jid, payload = store.claim(chunk, config, pricing)
                except Invalid as error:
                    if str(error) == "BUDGET_EXHAUSTED":
                        status = "WAIT_FOR_COLLECTION_OR_BUDGET_LIMIT"
                        break
                    raise
                write_once(
                    Path(output).parent / "batch-jobs" / jid / "submission.json",
                    {
                        "job_hash": jid,
                        "requested_model": config["model"],
                        "payload": payload,
                    },
                )
                # Any exception/no ID leaves the committed SUBMITTING reservation intact.
                raw = transport("POST", ENDPOINT, payload)
                store.acknowledge(jid, raw)
                write_once(
                    Path(output).parent / "batch-jobs" / jid / "acknowledged.json", raw
                )
                job_ids.add(jid)
                new_jobs.append(jid)
            if status == "WAIT_FOR_COLLECTION_OR_BUDGET_LIMIT":
                break
        assigned = sum(store.assigned(r["id"]) is not None for r in requests)
        return _summary(
            output,
            {
                "status": status,
                "requested_model": config["model"],
                "requests": len(requests),
                "assigned_requests": assigned,
                "new_batch_posts": len(new_jobs),
                "job_hashes": sorted(job_ids),
                "batch_ids": [store.job(j)["batch_id"] for j in sorted(job_ids)],
                "final_receipts_written": False,
                "next_action": "collect-batches; re-run execute only for unassigned requests if budget permits",
            },
        )
    finally:
        store.close()


def collect_batches(
    requests, manifest, config, pricing, approval, ledger_path, output, *, http=None
):
    _validate(requests, manifest, config, pricing, approval, ledger_path, collect=True)
    transport = http or (
        lambda method, url, payload=None: _http(config, method, url, payload)
    )
    store = Store(ledger_path, digest(manifest), approval)
    completed = []
    pending = []
    unassigned = []
    jobs = {}
    try:
        for req in requests:
            jid = store.assigned(req["id"])
            if jid is None:
                unassigned.append(req["id"])
            else:
                jobs[jid] = store.job(jid)
        for jid, job in sorted(jobs.items()):
            require(
                job["batch_id"] is not None
                and job["status"] not in {"SUBMITTING", "BLOCKED", "VIOLATION"},
                "UNCERTAIN_OR_FAILED_BATCH_NO_RESUBMISSION",
            )
            if job["status"] == "DONE":
                completed.extend(store.cached(jid))
                continue
            start = time.monotonic()
            raw = transport("GET", ENDPOINT + "/" + job["batch_id"])
            # GET failures keep the acknowledged ID and all reservations for later collection.
            store.observed(jid, raw)
            write_once(
                Path(output).parent
                / "batch-jobs"
                / jid
                / ("retrieval-" + digest(raw)[:16] + ".json"),
                raw,
            )
            if raw.get("status") not in TERMINAL:
                pending.append(job["batch_id"])
                continue
            try:
                receipts, failed = result_receipts(job, store.items(jid), raw)
                for r in receipts:
                    r["retrieval_seconds"] = time.monotonic() - start
                receipts = store.settle(jid, receipts, raw, failed=failed)
                require(not failed, "FAILED_BATCH_RESULTS_CHARGED_NO_SCIENTIFIC_OUTPUT")
                completed.extend(receipts)
            except Exception as error:
                if store.job(jid)["status"] not in {"DONE", "VIOLATION", "BLOCKED"}:
                    store.block(jid, error, raw)
                raise
        by_id = {r["request_id"]: r for r in completed}
        require(len(by_id) == len(completed), "DUPLICATE_SETTLED_RECEIPT")
        if not pending and not unassigned:
            require(
                set(by_id) == {r["id"] for r in requests}, "COLLECTED_REQUEST_COVERAGE"
            )
            write_jsonl_once(Path(output), [by_id[r["id"]] for r in requests])
        return _summary(
            output,
            {
                "status": "COMPLETED" if not pending and not unassigned else "PENDING",
                "completed_requests": len(completed),
                "pending_batch_ids": pending,
                "unassigned_requests": unassigned,
                "new_batch_posts": 0,
                "final_receipts_written": not pending and not unassigned,
            },
        )
    finally:
        store.close()
