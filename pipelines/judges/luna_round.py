"""Round K: Luna asked about one relation at a time, on already-scored DEV44 documents.

No training and no fitted parameter. Model-facing requests carry only the document text with
<ID text> markers, mention ids, surfaces and sentences; pair gold and ECB+ chains are read
only by scoring. Every variant is compared, on the same documents, with the frozen Luna x d1
fusion (threshold 0.39) built from the cached DEV44 controls.

experiment:
  luna_pair      one call per unordered mention pair: causal yes/no + probability
  luna_cascade   fusion band only: one yes/no call per relation type, eliminate types one call
                 at a time until two remain, then a contrastive final choice
  luna_proposer  one call per mention: "which events caused this one?"; proposals are added
  luna_event     Luna groups mentions into events, one call per event pair, copied to mentions
  luna_ensemble  cache-only replay of luna_pair + luna_event, frozen mean with Luna and d1
"""
import hashlib
import json
import re
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path

import yaml

from pipelines.jev.batch import BatchChatClient
from pipelines.jev.client import DecisionError
from pipelines.judges.common import FROZEN_THRESHOLD, Dev44, Scorer, compare, components, micro_f1

LUNA = {"openai/gpt-6-luna:batch", "openai/gpt-6-luna"}
SERVED = {"openai/gpt-6-luna", "openai/gpt-6-luna:batch", "openai/gpt-6-luna-20260922"}
EXPERIMENTS = {"luna_pair", "luna_cascade", "luna_proposer", "luna_event", "luna_ensemble"}


# ---------------------------------------------------------------- parsing

def last_object(text):
    """Last JSON object in the reply; ValueError when there is none."""
    decoder, found = json.JSONDecoder(), None
    for start in (i for i, ch in enumerate(text) if ch == "{"):
        try:
            value, _ = decoder.raw_decode(text[start:])
        except ValueError:
            continue
        if isinstance(value, dict):
            found = value
    if found is None:
        raise ValueError("no JSON object in the reply")
    return found


def task_of(body):
    match = re.match(r"\[task=(\w+)\]", body["messages"][0]["content"])
    if not match:
        raise ValueError("system prompt lacks a task tag")
    return match.group(1)


def parse(task, content, allowed=None):
    """Validated answer for one task; `allowed` restricts type names or mention ids."""
    obj = last_object(content)
    if task == "pair_decision":
        if not isinstance(obj.get("causal"), bool):
            raise ValueError("causal must be a boolean")
        p = obj.get("probability")
        if isinstance(p, bool) or not isinstance(p, (int, float)) or not 0 <= p <= 100:
            raise ValueError("probability must be in [0, 100]")
        return {"causal": obj["causal"], "probability": float(p) / 100, "direction": obj.get("direction")}
    if task == "type_check":
        if obj.get("answer") not in ("yes", "no"):
            raise ValueError("answer must be yes or no")
        return obj["answer"] == "yes"
    if task in ("type_eliminate", "type_final"):
        key = "eliminate" if task == "type_eliminate" else "choice"
        name = str(obj.get(key, "")).strip().upper()
        if allowed is not None and name not in allowed:
            raise ValueError(f"{key} {name!r} is not a remaining type")
        return name
    if task == "cause_proposer":
        ids = obj.get("causes")
        if not isinstance(ids, list):
            raise ValueError("causes must be a list")
        return sorted({str(i) for i in ids if allowed is None or str(i) in allowed})
    if task == "event_grouping":
        groups = obj.get("groups")
        if not isinstance(groups, list) or not all(isinstance(g, list) for g in groups):
            raise ValueError("groups must be a list of lists")
        return [sorted({str(m) for m in g if allowed is None or str(m) in allowed}) for g in groups]
    raise ValueError(f"unknown task {task}")


def _allowed(body):
    """Constraints recoverable from the request itself, so cached replies re-validate alike."""
    user = body["messages"][-1]["content"]
    task = task_of(body)
    if task in ("type_eliminate", "type_final"):
        return set(re.findall(r"^- ([A-Z_]+):", user, re.M))
    if task in ("cause_proposer", "event_grouping"):
        return set(re.findall(r"<(\d+) ", user.split("Text:\n", 1)[1]))
    return None


# ---------------------------------------------------------------- client

class LunaClient(BatchChatClient):
    """Luna through the OpenRouter batch API, replies validated per task before caching."""

    def __init__(self, *, cache_only=False, workers=1, output_price_per_million=.5, **kwargs):
        self.output_price = output_price_per_million
        kwargs["cache_path"] = str(Path(kwargs["cache_path"]).expanduser())
        super().__init__(**kwargs)
        if self.identity["model"] not in LUNA:
            raise ValueError(f"round K is pinned to {sorted(LUNA)}")
        self.batch = self.identity["model"].endswith(":batch")
        self.cache_only, self.workers = cache_only, workers
        self.lock_spend = threading.Lock()
        self.receipt = Counter()

    def validate_request(self, messages, questions):
        body, bound = super().validate_request(messages, questions)
        worst = (bound * self.input_price + self.max_output * self.output_price) / 1e6
        if self.mode != "mock" and worst > self.reservation:
            raise DecisionError(f"per-call reservation {self.reservation} below worst-case cost {worst:.4f}")
        if not self.batch:
            body["usage"] = {"include": True}
        return body, bound

    def _send(self, body):
        if self.mode != "mock":
            response = super()._send(body)
            usage = response.get("usage") or {}
            if usage.get("cost") is None and isinstance(usage.get("prompt_tokens"), int):
                # Known token price instead of an unknown cost, which the ledger would hold forever.
                usage["cost"] = (usage["prompt_tokens"] * self.input_price
                                 + int(usage.get("completion_tokens") or 0) * self.output_price) / 1e6
                response["usage"] = usage
            return response
        content = mock_answer(body)
        return {"model": "openai/gpt-6-luna", "provider": "mock", "usage": {"cost": 0},
                "choices": [{"finish_reason": "stop", "message": {"content": content}}]}

    def _validate_response(self, body, response):
        if not isinstance(response, dict) or response.get("model") not in SERVED | {"mock/" + self.identity["model_version"]}:
            raise DecisionError("unexpected chat model: " + str((response or {}).get("model")))
        choice = (response.get("choices") or [{}])[0]
        if choice.get("finish_reason") != "stop":
            raise DecisionError("incomplete chat generation: " + str(choice.get("finish_reason")))
        content = (choice.get("message") or {}).get("content")
        if not isinstance(content, str):
            raise DecisionError("chat response lacks text content")
        try:
            parse(task_of(body), content, _allowed(body))
        except ValueError as exc:
            raise DecisionError("unparseable answer: " + str(exc)[:300]) from exc

    def ask_many(self, requests):
        """`requests`: list of (system, user). Returns parsed answers in order (one batch)."""
        messages = [[{"role": "system", "content": s}, {"role": "user", "content": u}] for s, u in requests]
        if self.cache_only and self.mode != "mock":
            self._require_cached(messages)
        events = []
        if self.batch:
            packets = self.decide_many(messages, [events.append] * len(messages))
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                packets = list(pool.map(lambda m: self.decide(m, {}, on_attempt=events.append), messages))
        with self.lock_spend:
            for e in events:
                self.receipt["requests"] += 1
                self.receipt["cache_hits"] += int(e["cache_hit"])
                self.receipt["usd"] += e["cost_usd"] or 0
        out = []
        for m, packet in zip(messages, packets):
            body = {"messages": m}
            out.append(parse(task_of(body), packet["response"]["choices"][0]["message"]["content"], _allowed(body)))
        return out

    def _require_cached(self, messages):
        from benchmark.io import fingerprint
        for m in messages:
            if self.batch:
                key = fingerprint({**self.identity, "mode": self.mode, "messages": m,
                                   "reasoning_effort": self.reasoning_effort, "max_output_tokens": self.max_output})
            else:
                key = fingerprint({**self.identity, "mode": self.mode, "state": m, "questions": {},
                                   "expected_served_version": self.expected_version})
            if not self.cache.execute("SELECT 1 FROM decisions_v1 WHERE key=?", (key,)).fetchone():
                raise DecisionError("cache-only replay found an uncached request; no new call allowed")


def mock_answer(body):
    """Deterministic valid answers for offline tests; never used in live mode."""
    task, user = task_of(body), body["messages"][-1]["content"]
    h = int(hashlib.sha256(user.encode()).hexdigest(), 16)
    allowed = sorted(_allowed(body) or [])
    if task == "pair_decision":
        p = h % 101
        return f'ok\n{{"causal": {str(p >= 60).lower()}, "direction": "none", "probability": {p}}}'
    if task == "type_check":
        return '{"answer": "%s"}' % ("yes" if h % 3 == 0 else "no")
    if task == "type_eliminate":
        return '{"eliminate": "%s"}' % allowed[h % len(allowed)]
    if task == "type_final":
        return '{"choice": "%s"}' % allowed[h % len(allowed)]
    if task == "cause_proposer":
        return json.dumps({"causes": allowed[h % 7::9][:2]})
    if task == "event_grouping":
        return json.dumps({"groups": [allowed[i:i + 2] for i in range(0, len(allowed) - 1, 5)]})
    raise ValueError(task)


# ---------------------------------------------------------------- documents and prompts

class Docs:
    """DEV44 controls restricted to the configured documents, plus model-facing views."""

    def __init__(self, cfg):
        self.data = Dev44(cfg["controls"], cfg["suites_dir"])
        missing = set(cfg["documents"]) - set(self.data.docs)
        if missing:
            raise ValueError(f"documents not in DEV44: {sorted(missing)}")
        self.data.docs = sorted(cfg["documents"])
        self.sentences = {}
        for d in self.data.docs:
            raw = json.loads((Path(cfg["suites_dir"]) / f"{d}.json").read_text())["documents"][0]["sentences"]
            self.sentences[d] = raw if isinstance(raw, list) else json.loads(raw)
        self.text = {d: self.data.marked_text(d) for d in self.data.docs}

    def sentence(self, doc, mention):
        s = self.data.suite[doc]
        start, end = self.sentences[doc][int(s["sentence_of"][mention])]
        return " ".join(s["tokens"][start:end])

    def describe(self, doc, mentions):
        """One event: its mention(s), surface(s) and sentence(s)."""
        s = self.data.suite[doc]
        parts = [f'mention <{m}> "{s["surfaces"][m]}" in sentence: "{self.sentence(doc, m)}"' for m in mentions]
        return parts[0] if len(parts) == 1 else "the event mentioned as " + "; ".join(parts)

    def unordered(self, doc):
        return self.data.unordered(doc)


class Prompts:
    def __init__(self, path):
        self.raw = yaml.safe_load(Path(path).read_text())
        self.types = self.raw["types"]

    def __call__(self, task, **fields):
        spec = self.raw[task]
        return spec["system"].format(context=self.raw["context"].strip()), spec["user"].format(**fields)


# ---------------------------------------------------------------- experiments

def run_pairs(client, prompts, docs, doc, groups):
    """One pair_decision call per pair of groups; returns {(a, b): answer} over mention pairs."""
    members = {m: g for g in groups for m in g}
    gpairs = sorted({tuple(sorted((tuple(members[a]), tuple(members[b]))))
                     for a, b in docs.unordered(doc) if members[a] != members[b]})
    answers = client.ask_many([prompts("pair_decision", doc_text=docs.text[doc],
                                       event_a=docs.describe(doc, list(x)), event_b=docs.describe(doc, list(y)))
                               for x, y in gpairs])
    by_group = dict(zip(gpairs, answers))
    out = {}
    for a, b in docs.unordered(doc):
        if members[a] == members[b]:
            out[(a, b)] = {"causal": False, "probability": 0.0, "same_event": True}
        else:
            out[(a, b)] = by_group[tuple(sorted((tuple(members[a]), tuple(members[b]))))]
    return out, len(gpairs)


def exp_pair(client, prompts, docs, doc, cfg):
    singletons = [[m] for m in docs.data.mention_ids(doc)]
    answers, calls = run_pairs(client, prompts, docs, doc, singletons)
    return {"pairs": answers, "calls": calls}


def cascade_one(client, prompts, docs, doc, pairs):
    """Elimination cascade for a list of unordered pairs; one batch per stage."""
    types = prompts.types
    names = list(types)
    view = {p: dict(doc_text=docs.text[doc], event_a=docs.describe(doc, [p[0]]), event_b=docs.describe(doc, [p[1]]))
            for p in pairs}
    jobs = [(p, t) for p in pairs for t in names]
    answers = client.ask_many([prompts("type_check", question=types[t]["question"], **view[p]) for p, t in jobs])
    alive = {p: [t for (q, t), yes in zip(jobs, answers) if q == p and yes] for p in pairs}
    trace = {p: {"stage1_yes": list(alive[p]), "eliminated": []} for p in pairs}
    listing = lambda ts: "\n".join(f"- {t}: {types[t]['definition']}" for t in ts)
    while True:
        todo = [p for p in pairs if len(alive[p]) > 2]
        if not todo:
            break
        drop = client.ask_many([prompts("type_eliminate", candidates=listing(alive[p]), **view[p]) for p in todo])
        for p, t in zip(todo, drop):
            alive[p].remove(t)
            trace[p]["eliminated"].append(t)
    final_todo = [p for p in pairs if len(alive[p]) == 2]
    finals = client.ask_many([prompts("type_final", candidates=listing(alive[p]), **view[p]) for p in final_todo])
    chosen = dict(zip(final_todo, finals))
    out = {}
    for p in pairs:
        final = chosen.get(p) or (alive[p][0] if alive[p] else None)
        out[p] = {**trace[p], "final_type": final, "causal": bool(final and types[final]["causal"])}
    return out


def exp_cascade(client, prompts, docs, doc, cfg):
    low, high = cfg["band"]
    fused = docs.data.fused[doc]
    band = [p for p in docs.unordered(doc) if low <= fused[p] < high]
    out = cascade_one(client, prompts, docs, doc, band) if band else {}
    return {"pairs": {p: out.get(p) for p in docs.unordered(doc)}, "band_pairs": len(band)}


def exp_proposer(client, prompts, docs, doc, cfg):
    mentions = docs.data.mention_ids(doc)
    jobs = [(m, k) for k in range(cfg["samples"]) for m in mentions]
    answers = client.ask_many([
        (lambda s, u: (s, f"{u}\n(Sample {k}.)"))(*prompts("cause_proposer", doc_text=docs.text[doc],
                                                           event_a=docs.describe(doc, [m])))
        for m, k in jobs])
    votes = Counter()
    for (target, k), causes in zip(jobs, answers):
        for c in causes:
            if c != target:
                votes[(tuple(sorted((c, target))), k)] = 1
    per_pair = Counter()
    for (p, _), v in votes.items():
        per_pair[p] += v
    return {"pairs": {p: {"votes": per_pair.get(p, 0)} for p in docs.unordered(doc)}, "calls": len(jobs)}


def predicted_groups(client, prompts, docs, doc, samples, min_votes):
    mentions = docs.data.mention_ids(doc)
    requests = [(lambda s, u: (s, f"{u}\n(Sample {k}.)"))(*prompts("event_grouping", doc_text=docs.text[doc],
                                                                   mention_ids=", ".join(mentions)))
                for k in range(samples)]
    links = Counter()
    for groups in client.ask_many(requests):
        seen = set()
        for g in groups:
            seen |= {tuple(sorted(p)) for p in combinations(g, 2)}
        links.update(seen)
    kept = [p for p, v in links.items() if v >= min_votes]
    multi = components(kept, mentions)
    grouped = {m for g in multi for m in g}
    return multi + [[m] for m in mentions if m not in grouped]


def exp_event(client, prompts, docs, doc, cfg):
    groups = predicted_groups(client, prompts, docs, doc, cfg["grouping_samples"], cfg["grouping_min_votes"])
    answers, calls = run_pairs(client, prompts, docs, doc, groups)
    return {"pairs": answers, "calls": calls, "groups": [g for g in groups if len(g) > 1]}


RUNNERS = {"luna_pair": exp_pair, "luna_cascade": exp_cascade, "luna_proposer": exp_proposer,
           "luna_event": exp_event}


# ---------------------------------------------------------------- scoring

def p_causal(r):
    """Probability of a causal link, consistent with the boolean answer.

    K0 showed Luna often reports "probability" as confidence in its own answer
    ({"causal": false, "probability": 99}). Read it as confidence: a "no" maps below 0.5,
    a "yes" at or above 0.5. Label-free and fixed before any 10-document result.
    """
    p = r["probability"]
    return max(p, 1 - p) if r["causal"] else min(p, 1 - p)


def ordered(doc_pairs, value):
    """Symmetric per-ordered-pair scores from unordered results."""
    out = {}
    for (a, b), r in doc_pairs.items():
        out[(a, b)] = out[(b, a)] = value(r)
    return out


def baselines(docs):
    data = docs.data
    luna_sym = {d: {k: (r["luna"] + data.by_doc[d][(k[1], k[0])]["luna"]) / 2 for k, r in data.by_doc[d].items()}
                for d in data.docs}
    d1_sym = {d: {k: (r["d1"] + data.by_doc[d][(k[1], k[0])]["d1"]) / 2 for k, r in data.by_doc[d].items()}
              for d in data.docs}
    return {"fusion": (lambda d: data.fused[d], FROZEN_THRESHOLD),
            "luna_document_majority": (lambda d: {k: r["luna"] for k, r in data.by_doc[d].items()}, 4 / 7 - 1e-9),
            "d1_symmetric": (lambda d: d1_sym[d], .5)}, luna_sym, d1_sym


def systems(cfg, docs, results):
    """Pre-registered primary system and secondaries for the experiment."""
    data, exp = docs.data, cfg["experiment"]
    fused = data.fused
    fixed = lambda d, dec: {k: float(dec(d, k)) for k in data.keys[d]}
    if exp == "luna_pair":
        prob = {d: ordered(results[d]["pairs"], p_causal) for d in data.docs}
        yes = {d: ordered(results[d]["pairs"], lambda r: float(r["causal"])) for d in data.docs}
        mean = {d: {k: (fused[d][k] + prob[d][k]) / 2 for k in data.keys[d]} for d in data.docs}
        return {"primary:luna_pair_yes": (lambda d: yes[d], .5),
                "luna_pair_probability>=0.5": (lambda d: prob[d], .5),
                "mean(fusion,luna_pair_probability)>=0.39": (lambda d: mean[d], FROZEN_THRESHOLD)}
    if exp == "luna_cascade":
        def decision(d, k):
            r = results[d]["pairs"][tuple(sorted(k))]
            return r["causal"] if r is not None else fused[d][k] >= FROZEN_THRESHOLD
        def stage1(d, k):  # secondary: any causal type answered yes at stage 1
            r = results[d]["pairs"][tuple(sorted(k))]
            if r is None:
                return fused[d][k] >= FROZEN_THRESHOLD
            return any(docs_types[t]["causal"] for t in r["stage1_yes"])
        docs_types = cfg["_types"]
        return {"primary:fusion_with_cascade_in_band": (lambda d: fixed(d, decision), .5),
                "fusion_with_stage1_any_causal_yes": (lambda d: fixed(d, stage1), .5)}
    if exp == "luna_proposer":
        votes = {d: ordered(results[d]["pairs"], lambda r: r["votes"]) for d in data.docs}
        n = cfg["samples"]
        add = lambda v: (lambda d: fixed(d, lambda dd, k: fused[dd][k] >= FROZEN_THRESHOLD or votes[dd][k] >= v), .5)
        out = {f"primary:fusion_or_proposed>={cfg['min_votes']}/{n}": add(cfg["min_votes"])}
        for v in range(1, n + 1):
            out[f"fusion_or_proposed>={v}/{n}"] = add(v)
            out[f"proposer_alone>={v}/{n}"] = (lambda vv: (lambda d: fixed(d, lambda dd, k: votes[dd][k] >= vv), .5))(v)
        return out
    if exp == "luna_event":
        prob = {d: ordered(results[d]["pairs"], p_causal) for d in data.docs}
        yes = {d: ordered(results[d]["pairs"], lambda r: float(r["causal"])) for d in data.docs}
        mean = {d: {k: (fused[d][k] + prob[d][k]) / 2 for k in data.keys[d]} for d in data.docs}
        return {"primary:event_level_yes": (lambda d: yes[d], .5),
                "event_level_probability>=0.5": (lambda d: prob[d], .5),
                "mean(fusion,event_level_probability)>=0.39": (lambda d: mean[d], FROZEN_THRESHOLD)}
    raise ValueError(exp)


def ensemble_systems(docs, pair_results, event_results):
    data = docs.data
    _, luna_sym, d1_sym = baselines(docs)
    pp = {d: ordered(pair_results[d]["pairs"], p_causal) for d in data.docs}
    ep = {d: ordered(event_results[d]["pairs"], p_causal) for d in data.docs}
    py = {d: ordered(pair_results[d]["pairs"], lambda r: float(r["causal"])) for d in data.docs}
    ey = {d: ordered(event_results[d]["pairs"], lambda r: float(r["causal"])) for d in data.docs}
    mean4 = {d: {k: (luna_sym[d][k] + d1_sym[d][k] + pp[d][k] + ep[d][k]) / 4 for k in data.keys[d]} for d in data.docs}
    votes = {d: {k: (luna_sym[d][k] >= 4 / 7 - 1e-9) + (d1_sym[d][k] >= .5) + py[d][k] + ey[d][k] for k in data.keys[d]}
             for d in data.docs}
    return {"primary:mean(luna,d1,luna_pair,event_level)>=0.39": (lambda d: mean4[d], FROZEN_THRESHOLD),
            "majority>=3/4": (lambda d: votes[d], 3),
            "majority>=2/4": (lambda d: votes[d], 2)}


def evaluate(docs, named):
    scorer = Scorer(docs.data)
    base, _, _ = baselines(docs)
    control = scorer.fixed(base["fusion"][0], base["fusion"][1])
    out = {}
    for name, (score_of, threshold) in {**base, **named}.items():
        per = scorer.fixed(score_of, threshold)
        out[name] = compare(per, control, docs.data.docs)
        out[name]["per_document_f1"] = {d: round(100 * micro_f1([per[d]]), 2) for d in docs.data.docs}
    return out


def coref_quality(docs, results, cfg):
    from pipelines.judges.common import gold_chains, link_quality
    gold = gold_chains(set(docs.data.docs), cfg["repo_id"], cfg["revision"])
    return link_quality({d: results[d]["groups"] for d in docs.data.docs}, gold, docs.data.docs)


# ---------------------------------------------------------------- driver entry

def collect(cfg, make_client, prompts, docs, experiment, receipt):
    """One client per document thread: each owns its SQLite connections (not shareable across threads)."""
    results, lock = {}, threading.Lock()
    sub = {**cfg, "experiment": experiment}

    def one(doc):
        client = make_client()
        try:
            r = RUNNERS[experiment](client, prompts, docs, doc, sub)
        finally:
            with lock:
                receipt.update(client.receipt)
            client.close()
        with lock:
            results[doc] = r
            print("ROUNDK_DOC " + json.dumps({"experiment": experiment, "doc": doc, "receipt": dict(receipt)}),
                  flush=True)

    with ThreadPoolExecutor(max_workers=cfg["doc_workers"]) as pool:
        for future in [pool.submit(one, d) for d in docs.data.docs]:
            future.result()
    return results


def serializable(results):
    return {d: {k: ({f"{a},{b}": v for (a, b), v in r.items()} if k == "pairs" else r) for k, r in res.items()}
            for d, res in results.items()}


def run(cfg):
    docs, prompts = Docs(cfg), Prompts(cfg["prompts"])
    cfg = {**cfg, "_types": prompts.types}
    make = lambda: LunaClient(shared_budget=cfg["budget"], cache_only=cfg["experiment"] == "luna_ensemble",
                              workers=cfg["workers"], **cfg["client"])
    receipt = Counter()
    if cfg["experiment"] == "luna_ensemble":
        pair = collect(cfg, make, prompts, docs, "luna_pair", receipt)
        event = collect(cfg, make, prompts, docs, "luna_event", receipt)
        named, results = ensemble_systems(docs, pair, event), {"luna_pair": pair, "luna_event": event}
    else:
        results = collect(cfg, make, prompts, docs, cfg["experiment"], receipt)
        named = systems(cfg, docs, results)
    out = {"schema": "workere-round-k-result/v1", "experiment": cfg["experiment"],
           "documents": docs.data.docs, "role": "development comparison on already-scored DEV44 documents",
           "metrics": evaluate(docs, named), "receipt": dict(receipt)}
    if cfg["experiment"] == "luna_event":
        out["coref_link_quality_vs_ecb"] = coref_quality(docs, results, cfg)
    if cfg["experiment"] == "luna_cascade":
        finals = Counter(r["final_type"] for res in results.values() for r in res["pairs"].values() if r)
        out["cascade_final_types"] = dict(finals)
        out["band_pairs"] = sum(r["band_pairs"] for r in results.values())
    out["predictions"] = serializable(results) if cfg["experiment"] != "luna_ensemble" else None
    return out
