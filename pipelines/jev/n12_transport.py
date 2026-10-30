"""N12 one-shot batches keyed by logical generations, never by prompt bodies.

Inherited HTTP and idempotent GET polling only. No response cache is consulted,
and even byte-identical prompts occupy separate submitted request slots.
"""
import datetime
import json
import math
import pathlib
import time
import urllib.error

from pipelines.jev.batch import BatchChatClient, BATCH_ENDPOINT, _save


class FreshLogicalBatchClient(BatchChatClient):
    def decide_logical(self, rows, callback):
        if self.mode != "live" or not rows or len(rows) > 24:
            raise ValueError("N12 requires 1..24 live logical requests")
        identifiers = [r["custom_id"] for r in rows]
        if len(set(identifiers)) != len(rows):
            raise ValueError("logical custom IDs must be unique")
        bodies = [self.validate_request(r["body"]["messages"], {})[0] for r in rows]
        if any(body != row["body"] for body, row in zip(bodies, rows)):
            raise ValueError("prepared bodies differ from frozen requests")
        if len(rows) > self.max_calls or len(rows) * self.reservation > self.budget:
            raise ValueError("batch exceeds frozen logical allocation")
        self.batch_dir.mkdir(parents=True, exist_ok=True)
        receipt = self.batch_dir / "one-shot.json"
        if receipt.exists():
            raise RuntimeError("existing receipt: N12 prohibits automatic paid resubmission")
        payload = {
            "endpoint": "/v1/chat/completions",
            "model": self.identity["model"].removesuffix(":batch"),
            "completion_window": "24h",
            "requests": [
                {"custom_id": row["custom_id"],
                 "body": {k: v for k, v in body.items() if k != "model"}}
                for row, body in zip(rows, bodies)
            ],
        }
        _save(self.batch_dir / "submission-payload.json", payload)
        _save(receipt, {"status": "submitting", "logical_slots": identifiers,
                        "reservation_usd": len(rows) * self.reservation,
                        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})
        started = time.monotonic()
        try:
            batch = self._http("POST", BATCH_ENDPOINT, payload)
            if not isinstance(batch.get("id"), str) or not batch["id"]:
                raise ValueError("missing provider batch ID")
        except urllib.error.HTTPError as exc:
            _save(receipt, {"status": "rejected", "http_status": exc.code,
                            "no_paid_retry": True})
            raise RuntimeError("N12 POST rejected; no retry") from exc
        except Exception as exc:
            # The pre-POST receipt remains: an ambiguous POST is never repeated.
            raise RuntimeError("N12 POST outcome uncertain; no retry") from exc
        _save(self.batch_dir / "submission-ack.json", batch)
        _save(receipt, batch)
        print("N12_BATCH_SUBMITTED " + json.dumps({"id": batch["id"],
              "doc": rows[0]["doc"], "requests": len(rows),
              "distinct_body_hashes": len({r["body_sha256"] for r in rows})}), flush=True)
        batch = self._poll(batch, receipt)
        _save(self.batch_dir / "final-retrieval.json", batch)
        total_cost = (batch.get("usage") or {}).get("cost")
        if total_cost is not None and (type(total_cost) not in (float, int)
                or not math.isfinite(total_cost) or total_cost < 0):
            raise ValueError("invalid reported batch cost")
        results = batch.get("results") or []
        by_id = {r["custom_id"]: r for r in results}
        if len(by_id) != len(results) or set(by_id) != set(identifiers):
            raise ValueError("provider results do not cover logical slots exactly")
        for row, body in zip(rows, bodies):
            item = by_id[row["custom_id"]]
            response = (item.get("response") or {}).get("body")
            failure = None
            try:
                if item.get("error") or (item.get("response") or {}).get("status_code") != 200:
                    raise ValueError("failed request: " + str(item.get("error"))[:500])
                self._validate_response(body, response)
            except Exception as exc:
                failure = {"type": type(exc).__name__, "message": str(exc)[:1000]}
            callback({"logical_id": row["logical_id"], "custom_id": row["custom_id"],
                      "schedule": row["schedule"], "arm": row["arm"], "doc": row["doc"],
                      "slot": row["slot"], "demo": row["demo"],
                      "request": body, "response": response, "batch_id": batch["id"],
                      "cache_hit": False, "api_call": True, "error": failure,
                      "cost_usd": total_cost / len(rows) if total_cost is not None else None,
                      "cost_allocation": "equal_share_of_batch_total_not_measured_perrequest",
                      "provider_reported_batch_cost_usd": total_cost,
                      "latency_seconds": time.monotonic() - started})
        if batch.get("status") != "completed":
            raise RuntimeError("batch terminal status " + str(batch.get("status")))
        if total_cost is not None and total_cost > len(rows) * self.reservation + 1e-12:
            raise RuntimeError("provider cost exceeded conservative reservation")
        return {"doc": rows[0]["doc"], "status": "completed", "batch_id": batch["id"],
                "provider_reported_batch_cost_usd": total_cost, "logical_slots": len(rows)}
