"""Asynchronous OpenRouter batch transport with durable submission receipts.

A document's seven requests are submitted together. Existing batch IDs are polled
on resume; an ambiguous POST is never automatically submitted a second time.
"""
import fcntl
import json
import math
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from benchmark.io import fingerprint
from pipelines.jev.client import BudgetExceeded, DecisionError, _http_details, _server_wait
from pipelines.jev.llm import ChatClient

BATCH_ENDPOINT = "https://openrouter.ai/api/v1/batches"
TERMINAL = {"completed", "failed", "expired", "cancelled"}


def _save(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False))
    temp.replace(path)


class BatchChatClient(ChatClient):
    def __init__(self, **kwargs):
        self.batch_dir = Path(kwargs["cache_path"]).expanduser().parent / "luna-batch-jobs"
        super().__init__(**kwargs)

    def _http(self, method, url, payload=None):
        key = os.environ.get(self.key_env) or (os.environ.get("OPENROUTER") if self.key_env == "OPENROUTER_API_KEY" else None)
        if not key:
            raise DecisionError("missing credential: " + self.key_env)
        request = urllib.request.Request(url, method=method,
            data=json.dumps(payload, ensure_ascii=False).encode() if payload is not None else None,
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.load(response)

    def _poll(self, batch, manifest_path):
        deadline, previous, last_log = time.monotonic() + 25 * 3600, None, 0
        ticket = batch.get("budget_ticket")
        while batch.get("status") not in TERMINAL:
            if time.monotonic() > deadline:
                raise DecisionError("batch polling timeout; receipt retained for resumption")
            status = batch.get("status")
            if status != previous or time.monotonic() - last_log >= 60:
                print("LUNA_BATCH_PROGRESS " + json.dumps({"id": batch["id"], "status": status,
                    "request_counts": batch.get("request_counts")}), flush=True)
                previous, last_log = status, time.monotonic()
            time.sleep(30)
            try:
                batch = self._http("GET", BATCH_ENDPOINT + "/" + batch["id"])
            except urllib.error.HTTPError as exc:
                if exc.code not in {429, 500, 502, 503, 504}:
                    raise DecisionError(f"batch retrieval HTTP {exc.code}; receipt retained") from exc
                details = _http_details(exc)
                delay = max(30, _server_wait(details.get("headers", {}), time.time()))
                if delay > self.retry_max_wait:
                    raise DecisionError("provider batch polling cooldown exceeds configured bound") from exc
                time.sleep(delay)
                continue
            except (TimeoutError, urllib.error.URLError):
                continue  # polling an existing id is idempotent
            if ticket:
                batch["budget_ticket"] = ticket
            _save(manifest_path, batch)
        return batch

    def decide_many(self, messages, callbacks):
        if self.mode == "mock":
            return [self.decide(m, {}, on_attempt=callback) for m, callback in zip(messages, callbacks)]
        if not self.identity["model"].endswith(":batch"):
            raise ValueError("batch transport requires the requested :batch model identity")
        bodies = [self.validate_request(m, {})[0] for m in messages]
        keys = [fingerprint({**self.identity, "mode": self.mode, "messages": m,
                             "reasoning_effort": self.reasoning_effort, "max_output_tokens": self.max_output}) for m in messages]
        packets, pending = {}, {}
        for key, body, callback in zip(keys, bodies, callbacks):
            cached = self.cache.execute("SELECT response FROM decisions_v1 WHERE key=?", (key,)).fetchone()
            if cached:
                response = json.loads(cached[0])
                self._validate_response(body, response)
                callback({"cache_key": key, "cache_hit": True, "request": body, "response": response,
                          "cost_usd": 0, "latency_seconds": 0, "api_call": False})
                packets[key] = {"response": response, "cache_key": key, "cache_hit": True}
            else:
                pending[key] = body
        if not pending:
            return [packets[key] for key in keys]
        reserve = len(pending) * self.reservation
        if len(pending) > self.max_calls or reserve > self.budget:
            raise BudgetExceeded("batch exceeds the document's explicit call or paid budget")
        self.batch_dir.mkdir(parents=True, exist_ok=True)
        model = self.identity["model"].removesuffix(":batch")
        # Field order matters: the Batch API parses the routing header before requests.
        payload = {"endpoint": "/v1/chat/completions", "model": model, "completion_window": "24h",
                   "requests": [{"custom_id": key, "body": {k: v for k, v in pending[key].items() if k != "model"}}
                                for key in sorted(pending)]}
        path = self.batch_dir / (fingerprint(payload) + ".json")
        with path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if path.exists():
                batch = json.loads(path.read_text())
                if not batch.get("id"):
                    raise BudgetExceeded("previous batch submission has an uncertain outcome; inspect receipt before any resubmission")
                submitted = False
            else:
                ticket = self.shared_budget.reserve(reserve) if self.shared_budget else None
                _save(path, {"status": "submitting", "requested_model": self.identity["model"],
                             "requests": payload["requests"], "reservation_usd": reserve, "budget_ticket": ticket})
                try:
                    batch = self._http("POST", BATCH_ENDPOINT, payload)
                    if not isinstance(batch.get("id"), str):
                        raise ValueError("submission returned no batch id")
                except urllib.error.HTTPError as exc:
                    _save(path, {"status": "rejected", "http_status": exc.code})
                    raise DecisionError(f"batch submission HTTP {exc.code}; no sync fallback") from exc
                except Exception as exc:
                    raise BudgetExceeded("batch submission outcome uncertain; durable receipt prevents duplicate paid submission") from exc
                batch["budget_ticket"] = ticket
                _save(path, batch)
                submitted = True
        self.spent += reserve
        self.calls += len(pending) if submitted else 0
        print("LUNA_BATCH_SUBMITTED " + json.dumps({"id": batch["id"], "model": self.identity["model"],
            "reasoning_effort": self.reasoning_effort, "requests": len(pending), "new_submission": submitted,
            "receipt_path": str(path), "reservation_usd": reserve}), flush=True)
        start = time.monotonic()
        batch = self._poll(batch, path)
        if batch.get("status") != "completed":
            print("LUNA_BATCH_FAILED " + json.dumps(batch), flush=True)
            raise DecisionError("batch ended with status " + str(batch.get("status")))
        results = batch.get("results") or []
        by_id = {r["custom_id"]: r for r in results}
        if len(by_id) != len(results) or set(by_id) != set(pending):
            raise DecisionError("batch result ids do not cover exactly the submitted requests")
        total_cost = (batch.get("usage") or {}).get("cost")
        if total_cost is not None and (type(total_cost) not in (int, float) or not math.isfinite(total_cost) or total_cost < 0):
            raise DecisionError("invalid batch-reported cost")
        if self.shared_budget and batch.get("budget_ticket"):
            self.shared_budget.settle(batch["budget_ticket"], total_cost)
        allocated = total_cost / len(pending) if total_cost is not None else None
        errors = []
        for key, body, callback in zip(keys, bodies, callbacks):
            if key not in pending:
                continue
            item = by_id[key]
            response = (item.get("response") or {}).get("body")
            failure = None
            try:
                if item.get("error") or (item.get("response") or {}).get("status_code") != 200:
                    raise DecisionError("failed batch request: " + str(item.get("error"))[:500])
                self._validate_response(body, response)
            except Exception as exc:
                failure = {"type": type(exc).__name__, "message": str(exc)[:1000]}
                errors.append(failure)
            callback({"cache_key": key, "cache_hit": False, "request": body, "response": response,
                      "batch_id": batch["id"], "batch_api": True, "cost_usd": allocated,
                      "cost_allocation": "equal_share_of_reported_batch_total", "provider_reported_batch_cost_usd": total_cost,
                      "reserved_usd": self.reservation, "latency_seconds": time.monotonic() - start,
                      "api_call": submitted, "error": failure})
            if not failure:
                self.cache.execute("INSERT OR REPLACE INTO decisions_v1 VALUES (?,?,?)",
                    (key, json.dumps(body), json.dumps(response)))
                self.cache.commit()
                packets[key] = {"response": response, "cache_key": key, "cache_hit": False}
        self.spent += (total_cost - reserve) if total_cost is not None else 0
        print("LUNA_BATCH_COMPLETED " + json.dumps({"id": batch["id"], "requests": len(pending),
              "cost_usd": total_cost, "errors": len(errors)}), flush=True)
        if total_cost is not None and total_cost > reserve + 1e-12:
            raise BudgetExceeded("provider batch cost exceeded conservative reservation")
        if errors:
            raise DecisionError("batch has failed or malformed requests; document is not scored")
        return [packets[key] for key in keys]
