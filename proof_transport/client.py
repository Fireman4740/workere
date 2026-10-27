"""Opt-in OpenRouter client with durable reservations and NO automatic retries.

Importing/running offline stages cannot call this client. Live execution requires
an external approval file bound to the frozen plan/config and the public tariff
snapshot. API keys are read only at send time, never serialized.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import urllib.request
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from .pipeline import write_jsonl_once, write_once
from .schema import digest, require, text_hash


class Budget:
    """A failed/unknown request keeps its reservation: restarting cannot repay it."""

    def __init__(self, path: Path, scope: str, max_usd: float, max_tokens: int):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS scope (id TEXT PRIMARY KEY, max_usd REAL, max_tokens INTEGER)"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS request (id TEXT PRIMARY KEY, scope TEXT, usd REAL, tokens INTEGER, status TEXT, receipt TEXT)"
        )
        self.db.execute(
            "INSERT OR IGNORE INTO scope VALUES (?,?,?)", (scope, max_usd, max_tokens)
        )
        require(
            self.db.execute(
                "SELECT max_usd,max_tokens FROM scope WHERE id=?", (scope,)
            ).fetchone()
            == (max_usd, max_tokens),
            "BUDGET_SCOPE_CHANGED",
        )
        self.db.commit()
        self.scope, self.max_usd, self.max_tokens = scope, max_usd, max_tokens

    def reserve(self, rid: str, usd: float, tokens: int) -> dict | None:
        require(
            type(tokens) is int
            and tokens > 0
            and type(usd) in (float, int)
            and usd >= 0,
            "RESERVATION_VALUE",
        )
        self.db.execute("BEGIN IMMEDIATE")
        try:
            old = self.db.execute(
                "SELECT status,receipt,scope FROM request WHERE id=?", (rid,)
            ).fetchone()
            if old:
                require(old[2] == self.scope, "REQUEST_BUDGET_SCOPE_MISMATCH")
                require(old[0] == "DONE", "UNKNOWN_OR_FAILED_REQUEST_NO_AUTO_RETRY")
                self.db.rollback()
                return json.loads(old[1])
            require(
                not self.db.execute(
                    "SELECT 1 FROM request WHERE scope=? AND status='VIOLATION'",
                    (self.scope,),
                ).fetchone(),
                "UNRESOLVED_PROVIDER_BUDGET_VIOLATION",
            )
            spent, used = self.db.execute(
                "SELECT COALESCE(SUM(usd),0),COALESCE(SUM(tokens),0) FROM request WHERE scope=?",
                (self.scope,),
            ).fetchone()
            require(
                spent + usd <= self.max_usd and used + tokens <= self.max_tokens,
                "BUDGET_EXHAUSTED",
            )
            self.db.execute(
                "INSERT INTO request VALUES (?,?,?,?,?,NULL)",
                (rid, self.scope, usd, tokens, "RESERVED"),
            )
            self.db.commit()
            return None
        except Exception:
            if self.db.in_transaction:
                self.db.rollback()
            raise

    def settle(self, rid: str, receipt: dict) -> None:
        u = receipt["usage"]
        usd, tokens = u["cost_usd"], u["input_tokens"] + u["output_tokens"]
        self.db.execute("BEGIN IMMEDIATE")
        try:
            reserved = self.db.execute(
                "SELECT usd,tokens,status FROM request WHERE id=? AND scope=?",
                (rid, self.scope),
            ).fetchone()
            require(
                reserved is not None and reserved[2] == "RESERVED", "INVALID_SETTLEMENT"
            )
            # Always record the real spend, even if a provider violates our upper bound.
            violation = usd > reserved[0] or tokens > reserved[1]
            self.db.execute(
                "UPDATE request SET usd=?,tokens=?,status=?,receipt=? WHERE id=?",
                (
                    usd,
                    tokens,
                    "VIOLATION" if violation else "DONE",
                    json.dumps(receipt),
                    rid,
                ),
            )
            self.db.commit()
            require(not violation, "PROVIDER_EXCEEDED_RESERVATION_STOP")
        except Exception:
            if self.db.in_transaction:
                self.db.rollback()
            raise

    def close(self):
        self.db.close()


def tariff_snapshot(raw_text: str, model: str, fetched_utc: str | None = None) -> dict:
    payload = json.loads(raw_text)
    found = [m for m in payload["data"] if m["id"] == model]
    require(len(found) == 1, "MODEL_NOT_IN_TARIFF_CATALOG")
    row = found[0]
    required_parameters = {"max_tokens", "reasoning", "response_format"}
    require(
        required_parameters <= set(row.get("supported_parameters", [])),
        "UNSUPPORTED_MODEL_PARAMETERS",
    )
    p = row["pricing"]
    prices = [p] + p.get("overrides", [])
    prompt = max(Decimal(x["prompt"]) for x in prices)
    completion = max(Decimal(x["completion"]) for x in prices)
    require(prompt >= 0 and completion >= 0, "INVALID_PRICE")
    return {
        "schema": "eci-proof-pricing/v1",
        "url": "https://openrouter.ai/api/v1/models",
        "fetched_utc": fetched_utc or datetime.now(UTC).isoformat(),
        "raw_catalog_sha256_utf8": text_hash(raw_text),
        "model": model,
        "canonical_slug": row.get("canonical_slug"),
        "listed_pricing": p,
        "conservative_prompt_usd_per_token": str(prompt),
        "conservative_completion_usd_per_token": str(completion),
        "context_length": row["context_length"],
        "supported_parameters": row["supported_parameters"],
        "transport": "synchronous_no_batch_discount",
        "model_invocations": 0,
    }


def upper_bound(req: dict, pricing: dict) -> dict:
    # UTF-8 bytes plus deliberately generous chat framing overhead. No tokenizer
    # download or inferred cached-input discount. Actual usage is always settled.
    raw_bytes = len(json.dumps(req["messages"], ensure_ascii=False).encode())
    inp = raw_bytes + 512 * len(req["messages"]) + 2048
    out = req["max_output_tokens"]
    usd = Decimal(inp) * Decimal(
        pricing["conservative_prompt_usd_per_token"]
    ) + Decimal(out) * Decimal(pricing["conservative_completion_usd_per_token"])
    return {
        "input_token_upper_bound": inp,
        "output_token_upper_bound": out,
        "billed_token_upper_bound": inp + out,
        "usd_upper_bound": float(usd),
        "assumption": "at_most_one_token_per_UTF8_byte_plus_chat_framing; stop if provider exceeds",
    }


def validate_approval(
    approval: dict, manifest: dict, config: dict, pricing: dict
) -> None:
    require(
        set(approval)
        == {
            "approval",
            "approved_by",
            "plan_hash",
            "config_hash",
            "pricing_hash",
            "model",
            "max_usd",
            "max_billed_tokens",
            "expires_utc",
            "ledger_path",
        },
        "APPROVAL_SCHEMA",
    )
    require(
        approval["approval"] == "APPROVED_BY_USER"
        and bool(approval["approved_by"].strip()),
        "EXPLICIT_USER_APPROVAL_REQUIRED",
    )
    require(
        approval["plan_hash"] == digest(manifest)
        and approval["config_hash"] == digest(config)
        and approval["pricing_hash"] == digest(pricing)
        and approval["model"] == config["model"] == pricing["model"],
        "APPROVAL_BINDING",
    )
    require(
        0 < approval["max_usd"] <= config["max_usd"] <= 10
        and 0 < approval["max_billed_tokens"] <= config["max_billed_tokens"] <= 1500000,
        "APPROVAL_BUDGET",
    )
    require(
        pricing.get("canonical_slug") == "openai/" + config["expected_version"],
        "CATALOG_MODEL_VERSION_DRIFT",
    )
    expires = datetime.fromisoformat(approval["expires_utc"])
    require(
        expires.tzinfo is not None and expires > datetime.now(UTC),
        "APPROVAL_EXPIRED",
    )
    fetched = datetime.fromisoformat(pricing["fetched_utc"])
    age = (datetime.now(UTC) - fetched).total_seconds()
    require(0 <= age <= 86400, "STALE_PRICING_REFRESH_AND_REAPPROVE")


def execute(
    requests: list,
    manifest: dict,
    config: dict,
    pricing: dict,
    approval: dict,
    ledger_path: Path,
    output: Path,
    *,
    send=None,
) -> list:
    validate_approval(
        approval, manifest, config, pricing
    )  # before key lookup / any network
    require(
        Path(approval["ledger_path"]).is_absolute()
        and ledger_path.resolve() == Path(approval["ledger_path"]).resolve(),
        "APPROVED_LEDGER_PATH_REQUIRED",
    )
    require(
        config["endpoint"] == "https://openrouter.ai/api/v1/chat/completions",
        "ENDPOINT",
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
    key = os.environ.get(config["api_key_env"])
    if send is None:
        require(bool(key), "MISSING_API_KEY")
    receipts = []
    ledger = Budget(
        ledger_path,
        digest(manifest),
        approval["max_usd"],
        approval["max_billed_tokens"],
    )
    try:
        for r in requests:
            bound = upper_bound(r, pricing)
            require(
                bound["billed_token_upper_bound"] <= pricing["context_length"],
                "CONTEXT_LIMIT",
            )
            cached = ledger.reserve(
                r["id"], bound["usd_upper_bound"], bound["billed_token_upper_bound"]
            )
            if cached is not None:
                require(
                    "execution_error" not in cached,
                    "FAILED_RESPONSE_REQUIRES_TECHNICAL_REVIEW",
                )
                receipts.append(cached)
                continue
            payload = {
                "model": r["model"],
                "messages": r["messages"],
                "max_tokens": r["max_output_tokens"],
                "reasoning": {"effort": r["reasoning_effort"]},
                "response_format": {"type": "json_object"},
                "provider": {"allow_fallbacks": False, "require_parameters": True},
            }
            start = time.monotonic()
            if send is not None:
                raw = send(payload)
            else:
                request = urllib.request.Request(
                    config["endpoint"],
                    data=json.dumps(payload).encode(),
                    headers={
                        "Authorization": "Bearer " + key,
                        "Content-Type": "application/json",
                    },
                )
                with urllib.request.urlopen(request, timeout=600) as response:
                    raw = json.loads(response.read())
            # A parse/API failure retains the reservation. There are no hidden retries.
            write_once(output.parent / "raw" / (r["id"] + ".json"), raw)
            u = raw.get("usage", {})
            require(
                type(u.get("prompt_tokens")) is int
                and type(u.get("completion_tokens")) is int,
                "MISSING_BILLED_USAGE_STOP",
            )
            require(
                u["prompt_tokens"] >= 0 and u["completion_tokens"] >= 0,
                "INVALID_BILLED_USAGE",
            )
            billed_output = u["completion_tokens"]
            if "total_tokens" in u:
                require(
                    type(u["total_tokens"]) is int
                    and u["total_tokens"]
                    >= u["prompt_tokens"] + u["completion_tokens"],
                    "INVALID_TOTAL_BILLED_USAGE",
                )
                billed_output = u["total_tokens"] - u["prompt_tokens"]
            estimated = Decimal(u["prompt_tokens"]) * Decimal(
                pricing["conservative_prompt_usd_per_token"]
            ) + Decimal(billed_output) * Decimal(
                pricing["conservative_completion_usd_per_token"]
            )
            reported = Decimal(str(u["cost"])) if u.get("cost") is not None else None
            require(
                reported is None or reported.is_finite() and reported >= 0,
                "INVALID_PROVIDER_COST",
            )
            cost = float(reported if reported is not None else estimated)
            receipt = {
                "request_id": r["id"],
                "request_hash": digest(r),
                "model": r["model"],
                "mode": "live",
                "served_model": raw.get("model"),
                "expected_version": r["expected_version"],
                "version_verified": r["expected_version"] in str(raw.get("model", "")),
                "provider": raw.get("provider"),
                "response_id": raw.get("id"),
                "response": {},
                "usage": {
                    "input_tokens": u["prompt_tokens"],
                    "output_tokens": billed_output,
                    "cost_usd": cost,
                },
                "provider_reported_cost_usd": u.get("cost"),
                "provider_completion_tokens": u["completion_tokens"],
                "provider_total_tokens": u.get("total_tokens"),
                "cost_kind": "provider_reported"
                if reported is not None
                else "conservative_tariff_missing_provider_cost",
                "conservative_tariff_cost_usd": float(estimated),
                "seconds": time.monotonic() - start,
                "pricing_hash": digest(pricing),
                "system_fingerprint": raw.get("system_fingerprint"),
            }
            # Settle real cost even when JSON/finish_reason/model identity is invalid.
            try:
                choice = raw["choices"][0]
                require(
                    choice["finish_reason"] == "stop", "TRUNCATED_OR_ERROR_RESPONSE"
                )
                require(
                    raw.get("model")
                    in {
                        r["model"],
                        r["expected_version"],
                        "openai/" + r["expected_version"],
                    },
                    "UNEXPECTED_SERVED_MODEL",
                )
                receipt["response"] = json.loads(choice["message"]["content"])
            except Exception as error:
                receipt["execution_error"] = type(error).__name__ + ":" + str(error)
                ledger.settle(r["id"], receipt)
                write_once(output.parent / "failed" / (r["id"] + ".json"), receipt)
                raise
            ledger.settle(r["id"], receipt)
            receipts.append(receipt)
    finally:
        ledger.close()
        if receipts:
            # Whole-file export is only complete if all requests returned. Individual
            # receipts are durable in SQLite; never fabricate the missing responses.
            write_jsonl_once(
                output.parent
                / (output.stem + ".partial-" + digest(receipts)[:16] + ".jsonl"),
                receipts,
            )
    write_jsonl_once(output, receipts)
    return receipts
