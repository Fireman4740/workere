"""Budget and tariff guards for the opt-in batch-only ECI provider client.

No network on import. execute submits asynchronous jobs; collect only retrieves
acknowledged IDs. There is deliberately no synchronous execution/fallback path.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from .schema import digest, require, text_hash


class Budget:
    """Durable dollar AND token reservations shared by every stage of a plan."""

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
    require(
        {"max_tokens", "reasoning", "response_format"}
        <= set(row.get("supported_parameters", [])),
        "UNSUPPORTED_MODEL_PARAMETERS",
    )
    prices = [row["pricing"]] + row["pricing"].get("overrides", [])
    prompt = max(Decimal(x["prompt"]) for x in prices)
    completion = max(Decimal(x["completion"]) for x in prices)
    require(
        prompt.is_finite()
        and completion.is_finite()
        and prompt >= 0
        and completion >= 0,
        "INVALID_PRICE",
    )
    return {
        "schema": "eci-proof-pricing/v1",
        "url": "https://openrouter.ai/api/v1/models",
        "fetched_utc": fetched_utc or datetime.now(UTC).isoformat(),
        "raw_catalog_sha256_utf8": text_hash(raw_text),
        "model": model,
        "canonical_slug": row.get("canonical_slug"),
        "listed_pricing": row["pricing"],
        "conservative_prompt_usd_per_token": str(prompt),
        "conservative_completion_usd_per_token": str(completion),
        "context_length": row["context_length"],
        "supported_parameters": row["supported_parameters"],
        "transport": "batch_catalog"
        if model.endswith(":batch")
        else "synchronous_no_batch_discount",
        "model_invocations": 0,
    }


def upper_bound(req: dict, pricing: dict) -> dict:
    inp = (
        len(json.dumps(req["messages"], ensure_ascii=False).encode())
        + 512 * len(req["messages"])
        + 2048
    )
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
    approval: dict,
    manifest: dict,
    config: dict,
    pricing: dict,
    *,
    collection_only: bool = False,
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
        and isinstance(approval["approved_by"], str)
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
        type(approval["max_usd"]) in (int, float)
        and 0 < approval["max_usd"] <= config["max_usd"] <= 10
        and type(approval["max_billed_tokens"]) is int
        and 0 < approval["max_billed_tokens"] <= config["max_billed_tokens"] <= 1500000,
        "APPROVAL_BUDGET",
    )
    require(
        pricing.get("canonical_slug") == "openai/" + config["expected_version"],
        "CATALOG_MODEL_VERSION_DRIFT",
    )
    expires = datetime.fromisoformat(approval["expires_utc"])
    require(
        expires.tzinfo is not None and (collection_only or expires > datetime.now(UTC)),
        "APPROVAL_EXPIRED",
    )
    fetched = datetime.fromisoformat(pricing["fetched_utc"])
    age = (datetime.now(UTC) - fetched).total_seconds()
    require(
        age >= 0 and (collection_only or age <= 86400),
        "STALE_PRICING_REFRESH_AND_REAPPROVE",
    )


def execute(
    requests, manifest, config, pricing, approval, ledger_path, output, *, http=None
):
    """Submit only; never synchronously fall back or wait for generation."""
    from .batch import submit_batches

    return submit_batches(
        requests, manifest, config, pricing, approval, ledger_path, output, http=http
    )


def collect(
    requests, manifest, config, pricing, approval, ledger_path, output, *, http=None
):
    """GET existing IDs only, including after the original approval expires."""
    from .batch import collect_batches

    return collect_batches(
        requests, manifest, config, pricing, approval, ledger_path, output, http=http
    )
