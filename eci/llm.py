"""The chat client every LLM call goes through: inference, CoT synthesis and tool sub-calls.

What it fixes relative to agentere's LangGraph graph:

  * Truly concurrent. agentere's graph node was a SYNC function, which LangGraph runs on the
    event loop's default thread pool — min(32, cpu + 4) workers, 28 on this machine. Measured:
    200 concurrent 1-second calls took 8.2 s with at most 28 in flight, so `concurrency: 200`
    in a queue file delivered 28. Here calls are `ainvoke`d on the async HTTP client and
    `experiment.concurrency` is the real in-flight budget.
  * Retries what actually fails. agentere retried ValueError and asyncio.TimeoutError only;
    `openai.APIConnectionError`, `RateLimitError`, `InternalServerError` are neither, so one
    transient network error in one pass failed the whole document (all its passes) and scored
    it as an empty prediction. ChatOpenAI's own hidden retries are disabled (max_retries=0):
    every retry is ours, counted and printed.
  * Truncation is a failure. A generation that ends with finish_reason "length" or "error"
    (5.5% of one provider's long generations came back that way with HTTP 200) is retried
    instead of being parsed — a cut landing after an earlier complete array passed silently.
  * Every call is accounted. Tokens and provider-reported cost are summed live per role
    (inference / cot / tool), including tool sub-calls that agentere never traced, and every
    attempt — failed ones included — is one line of the run's trace file, tagged with its
    document and pass so the file is self-describing.
"""
from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import threading
import time
import urllib.request
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from langchain_core.caches import BaseCache
from langchain_core.load import dumps, loads
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import Generation


class TruncatedResponse(ValueError):
    pass


class CallFailed(RuntimeError):
    pass


def _retryable() -> tuple:
    import openai
    # ValueError covers ParseError / TruncatedResponse and langchain's malformed-response errors.
    return (ValueError, asyncio.TimeoutError, openai.APIError)


# ── Prompt-keyed response cache (CoT synthesis and tool sub-calls only) ───────

_CACHE_SCHEMA = """CREATE TABLE IF NOT EXISTS llm_cache (
    key TEXT PRIMARY KEY, model TEXT, created_at TEXT, generations TEXT)"""


class SqliteLLMCache(BaseCache):
    """Keyed on the serialized message list + LangChain's llm_string (model, temperature,
    reasoning effort, bound tools) + a namespace. Entries are stored with their token usage and
    cost stripped, so a replay costs nothing in the accounting. An undeserializable entry is
    dropped and treated as a miss. Inference never uses this cache."""

    def __init__(self, path: str, namespace: str = "") -> None:
        self.namespace = namespace
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(_CACHE_SCHEMA)
        self._conn.commit()

    def _key(self, prompt: str, llm_string: str) -> str:
        return sha256("\x00".join((self.namespace, llm_string, prompt)).encode("utf-8")).hexdigest()

    def lookup(self, prompt: str, llm_string: str):
        key = self._key(prompt, llm_string)
        with self._lock:
            row = self._conn.execute("SELECT generations FROM llm_cache WHERE key = ?", (key,)).fetchone()
        if row is None:
            self.misses += 1
            return None
        try:
            gens = [loads(blob) for blob in json.loads(row[0])]
        except Exception as e:
            print(f"[llm_cache] dropping undeserializable entry {key[:12]}: {e}")
            with self._lock:
                self._conn.execute("DELETE FROM llm_cache WHERE key = ?", (key,))
                self._conn.commit()
            self.misses += 1
            return None
        self.hits += 1
        return gens

    def update(self, prompt: str, llm_string: str, return_val: Sequence[Generation]) -> None:
        stored = []
        model = ""
        for gen in return_val:
            g = gen.model_copy(deep=True)  # the live object still goes to the caller untouched
            msg = getattr(g, "message", None)
            if msg is not None:
                msg.usage_metadata = None
                msg.response_metadata.pop("token_usage", None)
                msg.response_metadata["cache_hit"] = True
                model = model or msg.response_metadata.get("model_name", "")
            stored.append(g)
        blob = json.dumps([dumps(g) for g in stored])
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO llm_cache (key, model, created_at, generations) VALUES (?, ?, ?, ?)",
                (self._key(prompt, llm_string), model, datetime.now().isoformat(timespec="seconds"), blob))
            self._conn.commit()

    def clear(self, **kwargs) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM llm_cache")
            self._conn.commit()

    def stats(self) -> str:
        total = self.hits + self.misses
        return f"{self.hits} hits / {self.misses} misses" + (f" ({100 * self.hits / total:.0f}%)" if total else "")


# ── Accounting ────────────────────────────────────────────────────────────────

@dataclass
class RoleUsage:
    calls: int = 0
    failed_attempts: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    reasoning_tokens: int = 0
    reported_cost: float = 0.0
    seconds: float = 0.0


@dataclass
class Ledger:
    """Live usage per role. Replaces agentere's after-the-fact parse of the trace file, which
    missed every tool sub-call except eci_extractor's."""
    roles: Dict[str, RoleUsage] = field(default_factory=dict)
    errors: Dict[str, int] = field(default_factory=dict)

    def role(self, name: str) -> RoleUsage:
        return self.roles.setdefault(name, RoleUsage())

    def add(self, role: str, ai: AIMessage, seconds: float) -> None:
        u = self.role(role)
        u.calls += 1
        u.seconds += seconds
        meta = ai.usage_metadata or {}
        u.input_tokens += meta.get("input_tokens", 0) or 0
        u.output_tokens += meta.get("output_tokens", 0) or 0
        u.cache_read_tokens += (meta.get("input_token_details") or {}).get("cache_read", 0) or 0
        u.reasoning_tokens += (meta.get("output_token_details") or {}).get("reasoning", 0) or 0
        # An errored generation serializes token_usage as an explicit null.
        cost = ((ai.response_metadata or {}).get("token_usage") or {}).get("cost")
        if cost is not None:
            u.reported_cost += cost

    def fail(self, role: str, exc: BaseException) -> None:
        self.role(role).failed_attempts += 1
        name = type(exc).__name__
        self.errors[name] = self.errors.get(name, 0) + 1

    def totals(self) -> RoleUsage:
        t = RoleUsage()
        for u in self.roles.values():
            for k in vars(t):
                setattr(t, k, getattr(t, k) + getattr(u, k))
        return t

    def as_dict(self) -> dict:
        return {"roles": {k: vars(v) for k, v in self.roles.items()},
                "total": vars(self.totals()), "errors": dict(self.errors)}


_PRICES: Optional[Dict[str, tuple]] = None


def model_price(model_id: str, base_url: str) -> Optional[tuple]:
    """(USD per input token, per output token) from OpenRouter's public /models, fetched once."""
    global _PRICES
    if _PRICES is None:
        try:
            req = urllib.request.Request(base_url.rstrip("/") + "/models")
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
            _PRICES = {}
            for m in data.get("data", []):
                p = m.get("pricing") or {}
                try:
                    _PRICES[m["id"]] = (float(p["prompt"]), float(p["completion"]))
                except (KeyError, ValueError, TypeError):
                    pass
        except Exception as e:
            print(f"[llm] could not fetch pricing ({e}); cost will rely on provider reports only")
            _PRICES = {}
    return _PRICES.get(model_id) or _PRICES.get(model_id.split(":")[0])


def cost_usd(ledger: Ledger, model_id: str, base_url: str) -> Optional[float]:
    """Provider-reported cost when present (correct even when a run mixes models); otherwise an
    estimate from list prices, cache reads at 10% of the input price."""
    t = ledger.totals()
    if t.reported_cost > 0:
        return round(t.reported_cost, 6)
    price = model_price(model_id, base_url)
    if not price:
        return None
    pin, pout = price
    return round((t.input_tokens - t.cache_read_tokens) * pin + t.cache_read_tokens * pin * 0.1
                 + t.output_tokens * pout, 6)


class TraceWriter:
    """One JSON line per attempt: {meta..., ok, error, finish_reason, seconds, messages}.
    `messages` is agentere's format (LangChain `to_json()` of the full message list)."""

    def __init__(self, path: Optional[Path]):
        self.path = path
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict) -> None:
        if self.path is None:
            return
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")


# ── Client ────────────────────────────────────────────────────────────────────

def make_chat_model(*, model_id: str, temperature: float, reasoning_effort: Optional[str],
                    base_url: str, provider_order: Sequence[str], cache):
    """ChatOpenAI against OpenRouter. `provider` is an OpenRouter-only body field, so it rides in
    extra_body (model_kwargs would be splatted into the SDK's create() and rejected). A None
    reasoning_effort sends no field at all, so the provider's default applies."""
    from langchain_openai import ChatOpenAI
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    extra_body = {"provider": {"order": list(provider_order), "allow_fallbacks": True}} if provider_order else None
    return ChatOpenAI(model=model_id, temperature=temperature, reasoning_effort=reasoning_effort,
                      api_key=api_key, base_url=base_url, extra_body=extra_body, cache=cache,
                      max_retries=0)


@dataclass
class CallMeta:
    role: str                      # "inference" | "cot" | "tool:<name>"
    doc_id: str = ""
    pass_idx: Optional[int] = None
    block: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)   # extra MLflow trace tags


class ChatClient:
    """One configured chat model plus the run's accounting.

    `slots` is the run-wide in-flight budget, acquired per attempt (a retry goes to the back of
    the queue instead of holding a slot through its backoff). Tool sub-calls run while their
    orchestrator call holds a slot, so tool clients get their own budget or none — sharing the
    run's would deadlock once every slot is held by an orchestrator waiting on its tools.
    """

    def __init__(self, model, *, ledger: Ledger, trace: TraceWriter,
                 slots: Optional[asyncio.Semaphore], timeout: float, max_tool_rounds: int,
                 mlflow_traces: bool = False, cache: Optional[SqliteLLMCache] = None):
        self.model = model
        self.ledger = ledger
        self.trace = trace
        self.slots = slots
        self.timeout = timeout
        self.max_tool_rounds = max_tool_rounds
        self.mlflow_traces = mlflow_traces
        self.cache = cache

    async def _attempt(self, msgs: List[BaseMessage], tools, meta: CallMeta) -> None:
        """Run the tool loop, appending to `msgs` in place so that a failed attempt's trace
        still shows how far it got (the truncated answer, the tool results)."""
        bound = self.model.bind_tools(tools) if tools else self.model
        by_name = {t.name: t for t in tools or []}
        for _ in range(self.max_tool_rounds):
            t0 = time.monotonic()
            ai = await bound.ainvoke(msgs)
            self.ledger.add(meta.role, ai, time.monotonic() - t0)
            msgs.append(ai)
            finish = (ai.response_metadata or {}).get("finish_reason")
            if finish in ("length", "error"):
                raise TruncatedResponse(f"generation ended with finish_reason={finish!r}")
            if not tools or not ai.tool_calls:
                return
            results = await asyncio.gather(*[self._run_tool(by_name, tc) for tc in ai.tool_calls])
            msgs.extend(results)
        raise CallFailed(f"no final answer after {self.max_tool_rounds} tool rounds")

    @staticmethod
    async def _run_tool(by_name, tc) -> ToolMessage:
        tool = by_name.get(tc["name"])
        if tool is None:
            return ToolMessage(content=f"Error: unknown tool {tc['name']!r}", tool_call_id=tc["id"])
        try:
            out = await tool.ainvoke(tc["args"])
        except Exception as e:  # reported to the model, as LangGraph's ToolNode did
            out = f"Error: {type(e).__name__}: {e}"
        return ToolMessage(content=str(out), tool_call_id=tc["id"])

    def _span(self, meta: CallMeta):
        if not self.mlflow_traces:
            return nullcontext()
        import mlflow
        return mlflow.start_span(name=meta.role, span_type="CHAT_MODEL")

    async def once(self, messages: List[BaseMessage], meta: CallMeta, tools=None, attempt: int = 0,
                   validate: Optional[Callable[[AIMessage], Any]] = None):
        """One attempt: timeout-bounded, traced, accounted. Returns (messages, validated) and
        raises on any failure — `validate` raising included, so a response that arrived but
        does not parse is traced and counted as the failed attempt it is."""
        t0 = time.monotonic()
        record = {"role": meta.role, "doc_id": meta.doc_id, "pass": meta.pass_idx,
                  "block": meta.block, "attempt": attempt}
        msgs = list(messages)
        try:
            with self._span(meta) as span:
                await asyncio.wait_for(self._attempt(msgs, tools, meta), self.timeout)
                value = validate(msgs[-1]) if validate else None
                if span is not None:
                    import mlflow
                    span.set_inputs({"messages": [m.content for m in messages[-1:]]})
                    span.set_outputs({"content": msgs[-1].content})
                    mlflow.update_current_trace(tags={"doc_id": meta.doc_id, "role": meta.role,
                                                      **({"pass": str(meta.pass_idx)} if meta.pass_idx is not None else {}),
                                                      **meta.tags})
        except BaseException as e:
            if isinstance(e, Exception):
                self.ledger.fail(meta.role, e)
            self.trace.write({**record, "ok": False, "error": f"{type(e).__name__}: {e}",
                              "seconds": round(time.monotonic() - t0, 3),
                              "messages": [m.to_json() for m in msgs]})
            raise
        self.trace.write({**record, "ok": True, "error": None,
                          "finish_reason": (msgs[-1].response_metadata or {}).get("finish_reason"),
                          "seconds": round(time.monotonic() - t0, 3),
                          "messages": [m.to_json() for m in msgs]})
        return msgs, value

    async def call(self, messages: List[BaseMessage], meta: CallMeta, *, retries: int,
                   retry_wait: float, tools=None,
                   validate: Optional[Callable[[AIMessage], Any]] = None):
        """Up to `retries + 1` attempts. A ParseError from `validate` is retried like any other
        failure. Returns (messages, validated)."""
        retryable = _retryable()
        last: Optional[BaseException] = None
        for attempt in range(retries + 1):
            try:
                async with (self.slots if self.slots is not None else nullcontext()):
                    return await self.once(messages, meta, tools, attempt, validate)
            except retryable as e:
                last = e
                where = f"{meta.doc_id} pass {meta.pass_idx}" if meta.pass_idx is not None else meta.doc_id
                print(f"[{meta.role}] {where}: attempt {attempt} failed: {type(e).__name__}: {str(e)[:200]}")
                if retry_wait > 0 and attempt < retries:
                    await asyncio.sleep(retry_wait)
        raise CallFailed(f"{meta.role} {meta.doc_id}: {retries + 1} attempts failed; last: "
                         f"{type(last).__name__}: {last}")
