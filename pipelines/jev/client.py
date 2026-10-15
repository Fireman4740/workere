"""Typed Decisions client with SQLite cache, retries, rate and conservative spend reservations."""
from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from benchmark.io import fingerprint
from benchmark.schema import probability

OPENROUTER_ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
FREE_MODELS = {"inception/mercury-decide:free", "respan/span-01-lite", "respan/span-01-lite:free"}


class DecisionError(RuntimeError):
    pass


class BudgetExceeded(DecisionError):
    pass


def is_free(config):
    return (config.get("provider", "openrouter") == "openrouter"
            and config.get("endpoint", OPENROUTER_ENDPOINT) == OPENROUTER_ENDPOINT
            and config.get("model") in FREE_MODELS)


class DecisionClient:
    def __init__(self, *, model, model_version, provider="openrouter", endpoint=OPENROUTER_ENDPOINT,
                 api_key_env="OPENROUTER_API_KEY", mode="mock", cache_path="logs/jev-cache.sqlite",
                 timeout=60, retries=2, requests_per_minute=60, max_calls=100,
                 budget_usd=0, max_cost_per_call_usd=.002, input_price_per_million=.042,
                 max_context_tokens=32768, expected_served_version=None, transport=None):
        if mode not in {"mock", "live"} or not all(isinstance(v, str) and v for v in (model, model_version, provider, endpoint, api_key_env)):
            raise ValueError("invalid client identity or mode")
        for name, value in {"timeout": timeout, "requests_per_minute": requests_per_minute,
                            "max_cost_per_call_usd": max_cost_per_call_usd}.items():
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive")
        for name, value in {"budget_usd": budget_usd, "input_price_per_million": input_price_per_million}.items():
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be nonnegative")
        if any(type(v) is not int or v < 0 for v in (retries, max_calls)) or type(max_context_tokens) is not int or max_context_tokens <= 0:
            raise ValueError("invalid retry, call or context limit")
        self.identity = {"model": model, "model_version": model_version, "provider": provider, "endpoint": endpoint}
        self.mode, self.key_env, self.transport = mode, api_key_env, transport
        self.free = is_free({**self.identity})
        self.timeout, self.retries, self.rpm = timeout, retries, requests_per_minute
        self.max_calls, self.budget, self.reservation = max_calls, budget_usd, max_cost_per_call_usd
        self.input_price, self.max_context = input_price_per_million, max_context_tokens
        self.expected_version = expected_served_version
        self.calls, self.spent, self.next_request = 0, 0.0, 0.0
        self.lock = threading.RLock()
        self.key_locks = {}
        path = Path(cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.cache = sqlite3.connect(path, check_same_thread=False, timeout=60)
        self.cache.execute("PRAGMA journal_mode=WAL")
        self.cache.execute("CREATE TABLE IF NOT EXISTS decisions_v1 (key TEXT PRIMARY KEY, request TEXT NOT NULL, response TEXT NOT NULL)")
        self.cache.commit()

    def validate_request(self, state, questions):
        if not isinstance(state, str) or not state:
            raise ValueError("state must be a nonempty string")
        if not isinstance(questions, dict) or not questions:
            raise ValueError("questions must be a mapping")
        if any(q.get("type") != "noul" for q in questions.values()):
            raise ValueError("v1 Noul client requires typed noul questions")
        body = {"model": self.identity["model"], "state": state, "questions": questions}
        # UTF-8 bytes upper-bound text-token count conservatively, with a framing allowance.
        bound = len(json.dumps(body, ensure_ascii=False).encode()) + 1024
        if bound > self.max_context:
            raise DecisionError(f"context_overflow: conservative bound {bound} exceeds {self.max_context}; no truncation applied")
        if not self.free and bound * self.input_price / 1e6 > self.reservation:
            raise BudgetExceeded("per-call reservation is below the configured conservative input cost bound")
        return body, bound

    def _send(self, body):
        if self.transport is not None:
            return self.transport(body)
        if self.mode == "mock":
            p = (int(fingerprint(body)[:8], 16) % 10001) / 10000
            return {"model": "mock/" + self.identity["model_version"], "provider": "mock",
                    "answers": {key: {"type": "noul", "noul": p} for key in body["questions"]},
                    "usage": {"cost": 0}, "mock": True}
        key = os.environ.get(self.key_env)
        if not key and self.key_env == "OPENROUTER_API_KEY":
            key = os.environ.get("OPENROUTER")
        if not key:
            raise DecisionError(f"missing credential environment variable: {self.key_env}")
        request = urllib.request.Request(self.identity["endpoint"], data=json.dumps(body).encode(),
                                        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.load(response)

    def _validate_response(self, body, response):
        if not isinstance(response, dict) or not isinstance(response.get("model"), str):
            raise DecisionError("response lacks an actual served model identity")
        if self.mode != "mock" and self.expected_version and response["model"] != self.expected_version:
            raise DecisionError(f"unexpected served model version: {response['model']}")
        for key in body["questions"]:
            answer = response.get("answers", {}).get(key, {})
            if answer.get("type") != "noul" or answer.get("noul") is None:
                raise DecisionError(f"missing structured noul probability for {key}")
            probability(answer["noul"])

    def decide(self, state, questions, on_attempt=None):
        body, bound = self.validate_request(state, questions)
        key = fingerprint({**self.identity, "mode": self.mode, "state": state, "questions": questions,
                           "expected_served_version": self.expected_version})
        with self.lock:
            key_lock = self.key_locks.setdefault(key, threading.Lock())
        with key_lock:
            with self.lock:
                cached = self.cache.execute("SELECT response FROM decisions_v1 WHERE key=?", (key,)).fetchone()
            if cached:
                response = json.loads(cached[0])
                self._validate_response(body, response)
                event = {"cache_key": key, "cache_hit": True, "request": body, "response": response,
                         "cost_usd": 0, "latency_seconds": 0, "api_call": False}
                if on_attempt:
                    on_attempt(event)
                return {"response": response, "cache_key": key, "cache_hit": True}
            for attempt in range(self.retries + 1):
                reserve = 0 if self.free or self.mode == "mock" else self.reservation
                with self.lock:
                    if self.calls >= self.max_calls:
                        raise BudgetExceeded(f"maximum attempts reached ({self.max_calls})")
                    if self.spent + reserve > self.budget + 1e-12:
                        raise BudgetExceeded("explicit paid budget exhausted or absent")
                    self.spent += reserve
                    self.calls += 1
                    wait = max(0, self.next_request - time.monotonic()) if self.mode != "mock" else 0
                    self.next_request = time.monotonic() + wait + 60 / self.rpm
                if wait:
                    time.sleep(wait)
                start, cost, response, failure = time.perf_counter(), None, None, None
                try:
                    response = self._send(body)
                    cost = response.get("usage", {}).get("cost")
                    if cost is not None and (type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0):
                        raise DecisionError("invalid reported cost")
                    self._validate_response(body, response)
                    if self.free and cost not in (0, None):
                        raise BudgetExceeded("free route returned a nonzero cost")
                    if reserve and cost is not None:
                        with self.lock:
                            self.spent += cost - reserve
                            if cost > reserve + 1e-12:
                                raise BudgetExceeded("provider cost exceeded the configured per-attempt reservation")
                except Exception as exc:
                    failure = exc
                event = {"cache_key": key, "cache_hit": False, "request": body, "response": response,
                         "cost_usd": cost, "reserved_usd": reserve,
                         "latency_seconds": time.perf_counter() - start, "attempt": attempt,
                         "api_call": self.mode != "mock", "conservative_input_tokens": bound,
                         "error": {"type": type(failure).__name__, "message":
                                   f"HTTP {failure.code}" if isinstance(failure, urllib.error.HTTPError)
                                   else str(failure)[:1000]} if failure else None}
                if on_attempt:
                    on_attempt(event)
                if failure is None:
                    with self.lock:
                        self.cache.execute("INSERT OR IGNORE INTO decisions_v1 VALUES (?,?,?)",
                                           (key, json.dumps(body), json.dumps(response)))
                        self.cache.commit()
                    return {"response": response, "cache_key": key, "cache_hit": False}
                retry = isinstance(failure, (TimeoutError, urllib.error.URLError, DecisionError))
                if isinstance(failure, (BudgetExceeded,)) or (isinstance(failure, urllib.error.HTTPError)
                        and failure.code not in {429, 500, 502, 503, 504}):
                    retry = False
                if not retry or attempt == self.retries:
                    raise DecisionError(f"decision failed after {attempt + 1} attempt(s): "
                                        + event["error"]["message"]) from failure
                time.sleep(min(2 ** attempt, 8))
        raise AssertionError("unreachable")

    def close(self):
        self.cache.close()
