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
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

from benchmark.io import fingerprint
from benchmark.schema import probability

OPENROUTER_ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
FREE_MODELS = {"inception/mercury-decide:free", "respan/span-01-lite", "respan/span-01-lite:free"}
RETRY_HEADERS = {"retry-after", "x-ratelimit-limit", "x-ratelimit-remaining", "x-ratelimit-reset"}


class DecisionError(RuntimeError):
    pass


class BudgetExceeded(DecisionError):
    pass


def _http_details(error):
    """Keep only diagnostic rate headers and scalar error codes, never request headers."""
    out = {"http_status": error.code, "headers": {
        key.lower(): str(value)[:200] for key, value in (error.headers or {}).items()
        if key.lower() in RETRY_HEADERS}}
    try:
        body = json.loads(error.read(65536))
        detail = body.get("error", {})
        metadata = detail.get("metadata", {})
        out["provider_metadata"] = {key: metadata[key][:200] if isinstance(metadata[key], str) else metadata[key] for key in
            ("error_type", "reason", "limit_source", "provider_code", "provider_name")
            if isinstance(metadata.get(key), (str, int, float, bool))
            and (not isinstance(metadata[key], float) or math.isfinite(metadata[key]))}
    except (ValueError, TypeError, AttributeError, OSError):
        pass
    return out


def _date_seconds(value):
    try:
        date = parsedate_to_datetime(value)
    except (ValueError, TypeError, OverflowError):
        date = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return date.replace(tzinfo=date.tzinfo or timezone.utc).timestamp()


def _server_wait(headers, now):
    """Retry-After is seconds or an HTTP date; reset is an absolute epoch/date."""
    waits = [0.0]
    for key in ("retry-after", "x-ratelimit-reset"):
        value = headers.get(key)
        if value is None:
            continue
        try:
            try:
                number = float(value)
                if key == "x-ratelimit-reset":
                    number = number / 1000 if number > 1e12 else number
                    number -= now
            except ValueError:
                number = _date_seconds(value) - now
            if math.isfinite(number) and number >= 0:
                waits.append(number)
        except (ValueError, TypeError, OverflowError):
            continue
    return max(waits)


def is_free(config):
    return (config.get("provider", "openrouter") == "openrouter"
            and config.get("endpoint", OPENROUTER_ENDPOINT) == OPENROUTER_ENDPOINT
            and config.get("model") in FREE_MODELS)


class DecisionClient:
    def __init__(self, *, model, model_version, provider="openrouter", endpoint=OPENROUTER_ENDPOINT,
                 api_key_env="OPENROUTER_API_KEY", mode="mock", cache_path="logs/jev-cache.sqlite",
                 timeout=60, retries=2, requests_per_minute=60, max_calls=100,
                 budget_usd=0, max_cost_per_call_usd=.002, input_price_per_million=.042,
                 max_context_tokens=32768, expected_served_version=None, transport=None,
                 retry_backoff_seconds=1, retry_max_wait_seconds=86400):
        if mode not in {"mock", "live"} or not all(isinstance(v, str) and v for v in (model, model_version, provider, endpoint, api_key_env)):
            raise ValueError("invalid client identity or mode")
        for name, value in {"timeout": timeout, "requests_per_minute": requests_per_minute,
                            "max_cost_per_call_usd": max_cost_per_call_usd,
                            "retry_backoff_seconds": retry_backoff_seconds,
                            "retry_max_wait_seconds": retry_max_wait_seconds}.items():
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
        self.retry_backoff, self.retry_max_wait = retry_backoff_seconds, retry_max_wait_seconds
        if self.retry_backoff > self.retry_max_wait:
            raise ValueError("retry_backoff_seconds exceeds retry_max_wait_seconds")
        self.calls, self.spent, self.next_request = 0, 0.0, 0.0
        self.lock = threading.RLock()
        self.key_locks = {}
        path = Path(cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.cache = sqlite3.connect(path, check_same_thread=False, timeout=60)
        self.cache.execute("PRAGMA journal_mode=WAL")
        self.cache.execute("CREATE TABLE IF NOT EXISTS decisions_v1 (key TEXT PRIMARY KEY, request TEXT NOT NULL, response TEXT NOT NULL)")
        self.cache.execute("CREATE TABLE IF NOT EXISTS rate_slots_v1 (scope TEXT PRIMARY KEY, next_at REAL NOT NULL)")
        self.cache.commit()

    def _rate_slot(self):
        """Share provider pacing across clients/processes using this cache on one host."""
        scope = fingerprint({k: self.identity[k] for k in ("provider", "endpoint")})
        with self.lock:
            try:
                self.cache.execute("BEGIN IMMEDIATE")
                previous = self.cache.execute("SELECT next_at FROM rate_slots_v1 WHERE scope=?", (scope,)).fetchone()
                now = time.time()
                slot = max(now, previous[0] if previous else now)
                self.cache.execute("INSERT OR REPLACE INTO rate_slots_v1 VALUES (?,?)", (scope, slot + 60 / self.rpm))
                self.cache.commit()
            except BaseException:
                self.cache.rollback()
                raise
        return max(0, slot - now)

    def _defer_requests(self, seconds):
        """A provider cooldown also applies to other clients sharing this cache."""
        scope = fingerprint({k: self.identity[k] for k in ("provider", "endpoint")})
        with self.lock:
            try:
                self.cache.execute("BEGIN IMMEDIATE")
                previous = self.cache.execute("SELECT next_at FROM rate_slots_v1 WHERE scope=?", (scope,)).fetchone()
                next_at = max(time.time() + seconds, previous[0] if previous else 0)
                self.cache.execute("INSERT OR REPLACE INTO rate_slots_v1 VALUES (?,?)", (scope, next_at))
                self.cache.commit()
            except BaseException:
                self.cache.rollback()
                raise

    def validate_request(self, state, questions):
        if not isinstance(state, str) or not state:
            raise ValueError("state must be a nonempty string")
        if not isinstance(questions, dict) or not questions:
            raise ValueError("questions must be a mapping")
        if any(not isinstance(k, str) or not k for k in questions):
            raise ValueError("question IDs must be nonempty strings")
        for q in questions.values():
            if not isinstance(q, dict) or q.get("type") not in {"noul", "choice"}:
                raise ValueError("client requires typed noul or choice questions")
            if not isinstance(q.get("instructions"), str) or not q["instructions"].strip():
                raise ValueError("question instructions must be a nonempty string")
            criteria = q.get("criteria")
            if not isinstance(criteria, dict) or len(criteria) < 2 or not all(
                    isinstance(k, str) and isinstance(v, str) for k, v in criteria.items()):
                raise ValueError("criteria must map class names to plain strings")
            if q["type"] == "noul" and set(criteria) != {"true", "false"}:
                raise ValueError("noul criteria must be true and false")
            if q["type"] == "choice" and self.identity["model"].startswith("respan/span-"):
                raise ValueError("Span currently supports noul only; no implicit model substitution")
        body = {"model": self.identity["model"], "state": state, "questions": questions}
        # UTF-8 bytes upper-bound text-token count conservatively, with a framing allowance.
        bound = len(json.dumps(body, ensure_ascii=False).encode()) + 1024
        if bound > self.max_context:
            raise DecisionError(f"context_overflow: conservative bound {bound} exceeds {self.max_context}; no truncation applied")
        if self.mode != "mock" and not self.free and bound * self.input_price / 1e6 > self.reservation:
            raise BudgetExceeded("per-call reservation is below the configured conservative input cost bound")
        return body, bound

    def _send(self, body):
        if self.transport is not None:
            return self.transport(body)
        if self.mode == "mock":
            p = (int(fingerprint(body)[:8], 16) % 10001) / 10000
            answers = {}
            for key, q in body["questions"].items():
                if q["type"] == "noul":
                    answers[key] = {"type": "noul", "noul": p}
                else:
                    names = list(q["criteria"])
                    weights = [1 + int(hashlib.sha256((body["state"] + name).encode()).hexdigest()[:6], 16) for name in names]
                    probabilities = {name: w / sum(weights) for name, w in zip(names, weights)}
                    choice = max(probabilities, key=probabilities.get)
                    answers[key] = {"type": "choice", "choice": choice, "probabilities": probabilities,
                                    "confidence": probabilities[choice]}
            return {"model": "mock/" + self.identity["model_version"], "provider": "mock",
                    "answers": answers,
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
        for key, question in body["questions"].items():
            answer = response.get("answers", {}).get(key, {})
            if answer.get("type") != question["type"]:
                raise DecisionError(f"missing structured {question['type']} answer for {key}")
            if question["type"] == "noul":
                if answer.get("noul") is None:
                    raise DecisionError("missing positive probability")
                probability(answer.get("noul"))
            else:
                probs = answer.get("probabilities")
                if not isinstance(probs, dict) or set(probs) != set(question["criteria"]) or answer.get("choice") not in probs:
                    raise DecisionError("choice response does not match the requested classes")
                for value in probs.values():
                    if value is None:
                        raise DecisionError("missing class probability")
                    probability(value)
                if abs(sum(probs.values()) - 1) > 1e-4:
                    raise DecisionError("choice probabilities do not sum to one")
                if "confidence" in answer:
                    probability(answer["confidence"])

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
                wait = self._rate_slot() if self.mode != "mock" else 0
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
                details = _http_details(failure) if isinstance(failure, urllib.error.HTTPError) else {}
                retry = isinstance(failure, (TimeoutError, urllib.error.URLError, DecisionError))
                if isinstance(failure, BudgetExceeded) or (isinstance(failure, urllib.error.HTTPError)
                        and failure.code not in {429, 500, 502, 503, 504}):
                    retry = False
                retry = retry and attempt < self.retries
                delay = max(min(self.retry_backoff * 2 ** min(attempt, 30), self.retry_max_wait),
                            _server_wait(details.get("headers", {}), time.time())) if retry else None
                event = {"cache_key": key, "cache_hit": False, "request": body, "response": response,
                         "cost_usd": cost, "reserved_usd": reserve,
                         "latency_seconds": time.perf_counter() - start, "attempt": attempt,
                         "api_call": self.mode != "mock", "conservative_input_tokens": bound,
                         "rate_limit_wait_seconds": wait,
                         "retry_delay_seconds": delay,
                         "error": {"type": type(failure).__name__, "message":
                                   f"HTTP {failure.code}" if isinstance(failure, urllib.error.HTTPError)
                                   else str(failure)[:1000], **details} if failure else None}
                if on_attempt:
                    on_attempt(event)
                if failure is None:
                    with self.lock:
                        self.cache.execute("INSERT OR IGNORE INTO decisions_v1 VALUES (?,?,?)",
                                           (key, json.dumps(body), json.dumps(response)))
                        self.cache.commit()
                    return {"response": response, "cache_key": key, "cache_hit": False}
                if not retry:
                    raise DecisionError(f"decision failed after {attempt + 1} attempt(s): "
                                        + event["error"]["message"]) from failure
                if delay > self.retry_max_wait:
                    raise DecisionError(f"provider requires {delay:.0f}s cooldown, above the configured "
                                        f"{self.retry_max_wait:.0f}s limit; no early retry") from failure
                if details.get("http_status") == 429:
                    self._defer_requests(delay)
                print(f"decision retry={attempt + 1}/{self.retries} error={event['error']['message']} "
                      f"wait_seconds={delay:.1f}", flush=True)
                # Keep the worker interruptible during long quota cooldowns.
                remaining = delay
                while remaining > 0:
                    step = min(remaining, 60)
                    time.sleep(step)
                    remaining -= step
        raise AssertionError("unreachable")

    def close(self):
        self.cache.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
