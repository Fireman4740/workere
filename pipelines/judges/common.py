"""Shared pieces of the DeepSeek judge experiments on frozen DEV44 Luna x d1 signals.

Model-facing requests carry only document text with inline <ID text> mention markers and
mention ids. Annotations (pair gold, ECB+ chains) are read only by the scoring functions.
"""
import ast
import json
import math
import random
import re
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import groupby
from pathlib import Path

import numpy as np

from pipelines.jev.client import DecisionError
from pipelines.jev.llm import ChatClient

CHAT_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "deepseek/deepseek-v4.1-flash-1017"
FROZEN_THRESHOLD = .39
GRID = np.round(np.arange(.05, .96, .01), 2)


def _lit(value):
    return value if isinstance(value, (dict, list)) else ast.literal_eval(value)


# ---------------------------------------------------------------- data

class Dev44:
    """Cached frozen-fusion signals, one entry per ordered candidate pair."""

    def __init__(self, controls_path, suites_dir):
        self.by_doc = defaultdict(dict)
        for r in json.loads(Path(controls_path).read_text()):
            self.by_doc[r["doc"]][(r["a"], r["b"])] = r
        self.docs = sorted(self.by_doc)
        self.topic = {d: d.split("_")[0] for d in self.docs}
        self.suite = {}
        for d in self.docs:
            s = json.loads((Path(suites_dir) / f"{d}.json").read_text())["documents"][0]
            self.suite[d] = {k: _lit(s[k]) for k in ("tokens", "spans", "mentions", "surfaces", "sentence_of")}
        self.keys = {d: list(self.by_doc[d]) for d in self.docs}
        self.fused = {d: self._fused(d) for d in self.docs}

    def _fused(self, doc):
        """Frozen R3 rule mean_sym: average Luna vote and d1 probability per order, then both orders."""
        pairs, out = self.by_doc[doc], {}
        for (a, b), r in pairs.items():
            q = pairs.get((b, a), r)
            out[(a, b)] = ((r["luna"] + q["luna"]) / 2 + (r["d1"] + q["d1"]) / 2) / 2
        return out

    def mention_ids(self, doc):
        return [str(m) for m in self.suite[doc]["mentions"]]

    def marked_text(self, doc):
        """Same <ID text> convention as the Luna prompts, built from the frozen suite spans."""
        s = self.suite[doc]
        starts, ends = defaultdict(list), defaultdict(int)
        for mid, span in zip(s["mentions"], s["spans"]):
            span = sorted(span)
            for i in span:
                if i - 1 not in span:
                    starts[i].append(str(mid))
                if i + 1 not in span:
                    ends[i] += 1
        return " ".join("".join(f"<{m} " for m in starts.get(i, [])) + tok + ">" * ends.get(i, 0)
                        for i, tok in enumerate(s["tokens"]))

    def unordered(self, doc):
        return sorted({tuple(sorted(k)) for k in self.keys[doc]})


def gold_chains(docs, repo_id, revision):
    """ECB+ event chains as mention-id lists; scoring only, never model-facing."""
    from datasets import load_dataset
    out = {}
    for split in ("train", "dev"):
        for x in load_dataset(repo_id, split=split, revision=revision):
            if x["id"] in docs:
                out[x["id"]] = [[str(x["mentions"][i]) for i in c] for c in x["coref_clusters"]]
    missing = set(docs) - set(out)
    if missing:
        raise ValueError(f"gold chains missing for {sorted(missing)}")
    return out


# ---------------------------------------------------------------- model client

def last_json(text, kind):
    """Decode the last JSON array, respecting brackets inside JSON strings."""
    found = []
    decoder = json.JSONDecoder()
    for start in (i for i, ch in enumerate(text) if ch == "["):
        try:
            value, length = decoder.raw_decode(text[start:])
        except ValueError:
            continue
        if isinstance(value, kind):
            found.append((start + length, -start, value))
    if not found:
        raise ValueError("no JSON array found in the reply")
    return max(found, key=lambda f: (f[0], f[1]))[2]


class DeepSeekClient(ChatClient):
    """OpenRouter chat completions for one pinned DeepSeek model, priced from usage tokens.

    `parse(content, body)` must raise ValueError on an unusable answer; the call is then
    retried and never scored as a default.
    """

    def __init__(self, *, parse, output_price_per_million, **kwargs):
        self.parse, self.output_price = parse, output_price_per_million
        kwargs.setdefault("endpoint", CHAT_ENDPOINT)
        kwargs["cache_path"] = str(Path(kwargs["cache_path"]).expanduser())
        super().__init__(**kwargs)
        if self.identity["model"] != MODEL:
            raise ValueError(f"judge model is pinned to {MODEL}")
        # DecisionClient's cache fingerprints identity + messages, rather than body.
        # Include every request knob so changing reasoning cannot reuse a weaker run.
        self.identity.update(reasoning_effort=self.reasoning_effort,
                             max_output_tokens=self.max_output, judge_request_version=2)

    def validate_request(self, messages, questions):
        body = {"model": self.identity["model"], "messages": messages, "max_tokens": self.max_output,
                "reasoning": {"effort": self.reasoning_effort}, "usage": {"include": True},
                "provider": {"require_parameters": True}}
        bound = len(json.dumps(body, ensure_ascii=False).encode()) + 1024
        if bound + self.max_output > self.max_context:
            raise DecisionError("chat context overflow; no truncation applied")
        worst = (bound * self.input_price + self.max_output * self.output_price) / 1e6
        if self.mode != "mock" and worst > self.reservation:
            raise DecisionError(f"per-call reservation {self.reservation} below worst-case cost {worst:.4f}")
        return body, bound

    def _send(self, body):
        if self.mode == "mock":
            return super()._send(body)
        response = super(ChatClient, self)._send(body)
        usage = response.get("usage") or {}
        if usage.get("cost") is None and isinstance(usage.get("prompt_tokens"), int):
            usage["cost"] = (usage["prompt_tokens"] * self.input_price
                             + int(usage.get("completion_tokens") or 0) * self.output_price) / 1e6
            response["usage"] = usage
        return response

    def _validate_response(self, body, response):
        if not isinstance(response, dict) or not isinstance(response.get("model"), str):
            raise DecisionError("chat response lacks served model")
        if self.mode != "mock" and not response["model"].startswith("deepseek/deepseek-v4.1-flash"):
            raise DecisionError("unexpected chat model: " + response["model"])
        choice = (response.get("choices") or [{}])[0]
        if choice.get("finish_reason") != "stop":
            raise DecisionError("incomplete chat generation: " + str(choice.get("finish_reason")))
        content = (choice.get("message") or {}).get("content")
        if not isinstance(content, str):
            raise DecisionError("chat response lacks text content")
        try:
            self.parse(content, body)
        except ValueError as exc:
            raise DecisionError("unparseable answer: " + str(exc)[:300]) from exc

    def ask(self, system, user, sample):
        """`sample` enters the request so repeated samples are distinct cache entries."""
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": f"{user}\n\n(Sample {sample}.)"}]
        attempts = []

        def record(event):
            attempts.append(event)
            response = event.get("response") or {}
            print("JUDGE_ATTEMPT " + json.dumps({
                "cache_key": event["cache_key"], "cache_hit": event["cache_hit"],
                "attempt": event.get("attempt"), "requested_reasoning_effort": self.reasoning_effort,
                "served_model": response.get("model"), "provider": response.get("provider"),
                "usage": response.get("usage"), "cost_usd": event["cost_usd"],
                "reserved_usd": event.get("reserved_usd"), "error": event.get("error")}), flush=True)

        out = self.decide(messages, None, on_attempt=record)
        response = out["response"]
        return {"content": response["choices"][0]["message"]["content"], "cache_hit": out["cache_hit"],
                "cost_usd": sum(e["cost_usd"] if e["cost_usd"] is not None else e.get("reserved_usd", 0)
                                for e in attempts),
                "api_attempts": sum(e["api_call"] for e in attempts),
                "usage": response.get("usage")}


def client_from(cfg, parse):
    c = dict(cfg["client"])
    return DeepSeekClient(parse=parse, shared_budget=cfg["budget"], **c)


def run_parallel(jobs, workers):
    """Cancel queued work on failure; let only the already running calls settle."""
    pool = ThreadPoolExecutor(max_workers=workers)
    futures = {pool.submit(job): i for i, job in enumerate(jobs)}
    out = [None] * len(futures)
    try:
        for future in as_completed(futures):
            out[futures[future]] = future.result()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    return out


def chunks(items, size):
    return [items[i:i + size] for i in range(0, len(items), size)]


class Spend:
    """Thread-safe receipts; the SQLite ledger is the enforcement authority."""
    def __init__(self):
        self.lock = threading.Lock()
        self.value = {"usd": 0.0, "calls": 0, "api_attempts": 0, "cache_hits": 0}

    def add(self, reply):
        with self.lock:
            self.value["usd"] += reply["cost_usd"]
            self.value["calls"] += 1
            self.value["api_attempts"] += reply["api_attempts"]
            self.value["cache_hits"] += int(reply["cache_hit"])


# ---------------------------------------------------------------- scoring

def components(links, members):
    """Connected components (size >= 2) of an undirected link set."""
    parent = {m: m for m in members}

    def find(m):
        while parent[m] != m:
            parent[m] = parent[parent[m]]
            m = parent[m]
        return m
    for a, b in links:
        parent[find(a)] = find(b)
    groups = defaultdict(list)
    for m in members:
        groups[find(m)].append(m)
    return [sorted(g) for g in groups.values() if len(g) > 1]


def chain_links(chains):
    return {tuple(sorted((a, b))) for c in chains for a in c for b in c if a < b}


def link_quality(pred, gold, docs):
    tp = sum(len(chain_links(pred[d]) & chain_links(gold[d])) for d in docs)
    np_ = sum(len(chain_links(pred[d])) for d in docs)
    ng = sum(len(chain_links(gold[d])) for d in docs)
    p, r = (tp / np_ if np_ else None), (tp / ng if ng else None)
    return {"precision": p, "recall": r, "true_links": tp, "predicted_links": np_, "gold_links": ng}


def propagate(scores, chains):
    """Max cluster-pair score over members; pairs inside one cluster are not causal."""
    cid = {}
    for i, c in enumerate(chains):
        for m in c:
            cid.setdefault(m, f"c{i}")
    cluster = lambda m: cid.get(m, f"m{m}")
    best = defaultdict(float)
    for (a, b), s in scores.items():
        if cluster(a) != cluster(b):
            key = tuple(sorted((cluster(a), cluster(b))))
            best[key] = max(best[key], s)
    return {(a, b): 0.0 if cluster(a) == cluster(b) else best[tuple(sorted((cluster(a), cluster(b))))]
            for (a, b) in scores}


class Scorer:
    def __init__(self, data):
        self.data = data
        self.gold = {d: np.array([data.by_doc[d][k]["gold"] for k in data.keys[d]], bool) for d in data.docs}

    def arr(self, doc, scores):
        return np.array([scores[k] for k in self.data.keys[doc]], float)

    def counts(self, doc, scores, threshold):
        p, g = self.arr(doc, scores) >= threshold, self.gold[doc]
        return {"tp": int((p & g).sum()), "fp": int((p & ~g).sum()), "fn": int((~p & g).sum())}

    def grid(self, doc, scores):
        p, g = self.arr(doc, scores)[None, :] >= GRID[:, None], self.gold[doc]
        return np.stack([(p & g).sum(1), (p & ~g).sum(1), (~p & g).sum(1)], 1)

    def lto(self, variants, thresholds=None):
        """Leave-one-topic-out choice over named score variants (and the threshold grid when
        `thresholds` is None). Returns per-document counts and the per-topic choices."""
        docs, topic = self.data.docs, self.data.topic
        if thresholds is None:
            table = {name: {d: self.grid(d, f(d)) for d in docs} for name, f in variants.items()}
        else:
            table = {name: {d: np.array([[self.counts(d, f(d), thresholds)[k] for k in ("tp", "fp", "fn")]])
                            for d in docs} for name, f in variants.items()}
        per_doc, choices = {}, {}
        for t in sorted(set(topic.values()), key=int):
            train = [d for d in docs if topic[d] != t]
            best = None
            for name in variants:
                f1 = f1_arr(sum(table[name][d] for d in train))
                i = int(np.argmax(f1))
                if best is None or f1[i] > best[0] + 1e-12:
                    best = (f1[i], name, i)
            _, name, i = best
            th = float(GRID[i]) if thresholds is None else thresholds
            choices[t] = {"variant": name, "threshold": th}
            for d in docs:
                if topic[d] == t:
                    per_doc[d] = self.counts(d, variants[name](d), th)
        return per_doc, choices

    def fixed(self, score_of, threshold=FROZEN_THRESHOLD):
        return {d: self.counts(d, score_of(d), threshold) for d in self.data.docs}

    def ap(self, score_of):
        rows = sorted(((score_of(d)[k], bool(self.data.by_doc[d][k]["gold"])) for d in self.data.docs
                       for k in self.data.keys[d]), key=lambda x: -x[0])
        pos, hits, total = sum(g for _, g in rows), 0, 0.0
        seen = 0
        for _, tied in groupby(rows, key=lambda row: row[0]):
            tied = list(tied)
            new = sum(g for _, g in tied)
            hits += new
            seen += len(tied)
            total += new * hits / seen
        return total / pos if pos else None


def f1_arr(c):
    tp, fp, fn = c[..., 0], c[..., 1], c[..., 2]
    den = 2 * tp + fp + fn
    return np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)


def micro_f1(counts):
    tp, fp, fn = (sum(c[k] for c in counts) for k in ("tp", "fp", "fn"))
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def compare(per, base, docs, samples=2000, seed=42):
    """Micro-F1 of `per`, delta vs `base` in points, paired document bootstrap 95% CI."""
    rng = random.Random(seed)
    draws = [[rng.choice(docs) for _ in docs] for _ in range(samples)]
    deltas = sorted(100 * (micro_f1([per[d] for d in s]) - micro_f1([base[d] for d in s])) for s in draws)
    totals = {k: sum(per[d][k] for d in docs) for k in ("tp", "fp", "fn")}
    return {"f1": round(100 * micro_f1(list(per.values())), 2), **totals,
            "delta_points": round(100 * (micro_f1(list(per.values())) - micro_f1(list(base.values()))), 2),
            "ci95": [round(deltas[int(.025 * samples)], 2), round(deltas[int(.975 * samples) - 1], 2)],
            "p_gain": sum(x > 0 for x in deltas) / samples}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def mention_pattern(text):
    return set(re.findall(r"<(\d+) ", text))
