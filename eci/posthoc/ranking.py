"""Resampling votes as a cause-retrieval ranker (agentere: scripts/experiments/vote_ranking.py).

No threshold: the score of candidate cause c for effect e is the number of passes that
predicted c -> e.

  query     every event annotated as the effect of >= 1 gold causal link ("directed"), or every
            event in a gold pair with partners in either direction ("undirected" — the setting
            to trust on ESC, whose FALLING_ACTION gold points the wrong way half the time)
  items     every other mention of the document (unvoted = 0)
  metrics   Hit@k (a relevant item in the top k) for k = 1..K, and MRR

Ties break by a seeded shuffle, so a scorer that scores everything alike is the random
baseline. Baselines rank the same items by cosine between "trigger: sentence" strings: TF-IDF
and sentence embeddings. None reads gold at prediction time.

Per-pass edges: from a workere trace (each line names its document and pass) a pass votes at
most once per edge even when it states the fact twice ("a,b causes" + "b,a causedby"). An
agentere report has only per-pair tallies, so there an edge's votes are max(tally of (a,b)
"causes", tally of (b,a) "causedby") — a lower bound on the passes asserting it.
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from typing import Callable, Dict, List

from eci.parse import ParseError, parse_triples
from eci.posthoc.runs import RunVotes


def edge_votes(run: RunVotes) -> Dict[str, Counter]:
    """{doc: Counter({(cause, effect): passes})}."""
    trace = run.path.with_name(run.name + ".traces.jsonl")
    if run.framework == "workere" and trace.exists():
        per_pass: Dict[tuple, set] = {}
        for line in trace.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec["role"] != "inference" or not rec["ok"]:
                continue
            try:
                triples = parse_triples(rec["messages"][-1]["kwargs"]["content"])
            except ParseError:
                continue
            edges = set()
            for a, lbl, b in triples:
                if lbl == "causes":
                    edges.add((a, b))
                elif lbl == "causedby":
                    edges.add((b, a))
            per_pass[(rec["doc_id"], rec["pass"], rec["block"])] = edges
        out: Dict[str, Counter] = defaultdict(Counter)
        for (doc, _, _), edges in per_pass.items():
            out[doc].update(edges)
        return out
    out = defaultdict(Counter)
    for doc_id, d in run.docs.items():
        for (a, b), c in d.votes.items():
            out[doc_id][(a, b)] = max(out[doc_id][(a, b)], c.get("causes", 0))
            out[doc_id][(b, a)] = max(out[doc_id][(b, a)], c.get("causedby", 0))
    return out


def gold_edges(doc) -> set:
    return {(a, b) if l == "causes" else (b, a) for a, l, b in doc.gold if l in ("causes", "causedby")}


def queries(doc, directed: bool):
    rel = defaultdict(set)
    for c, e in gold_edges(doc):
        rel[e].add(c)
        if not directed:
            rel[c].add(e)
    return [(q, [m for m in doc.mentions if m != q], rel[q]) for q in doc.mentions if rel[q]]


def event_texts(doc) -> Dict[str, str]:
    out = {}
    for m in doc.mentions:
        trig = doc.surfaces.get(m, m)
        s = doc.sentence_of.get(m)
        out[m] = f"{trig}: {' '.join(doc.tokens[doc.sentences[s][0]:doc.sentences[s][1]])}" if s is not None else trig
    return out


def _cosine_scorer(docs, encode) -> Callable:
    sims, texts, spans = {}, [], []
    for d in docs:
        t = event_texts(d)
        spans.append((d.id, list(t)))
        texts += list(t.values())
    X = encode(texts)
    row = 0
    for doc_id, ids in spans:
        block = X[row:row + len(ids)]
        S = block @ block.T
        sims[doc_id] = ({m: i for i, m in enumerate(ids)}, S.toarray() if hasattr(S, "toarray") else S)
        row += len(ids)

    def score(doc, q, c):
        idx, S = sims[doc.id]
        return float(S[idx[q], idx[c]])
    return score


def tfidf(docs):
    from sklearn.feature_extraction.text import TfidfVectorizer
    return _cosine_scorer(docs, lambda texts: TfidfVectorizer().fit_transform(texts))


def embedding(docs, model_name: str):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device="cpu")
    return _cosine_scorer(docs, lambda texts: model.encode(texts, convert_to_numpy=True,
                                                           normalize_embeddings=True, batch_size=64))


def evaluate(docs, score: Callable, directed: bool, k_max: int, seed: int) -> dict:
    rng = random.Random(seed)
    hits = [0] * k_max
    rr = n = n_items = 0
    for d in docs:
        for q, cands, relevant in queries(d, directed):
            order = cands[:]
            rng.shuffle(order)
            order.sort(key=lambda c: -score(d, q, c))
            first = next(i for i, c in enumerate(order) if c in relevant)
            for k in range(first, k_max):
                hits[k] += 1
            rr += 1.0 / (first + 1)
            n += 1
            n_items += len(cands)
    if not n:
        return {"hit": [0.0] * k_max, "mrr": 0.0, "n_queries": 0, "mean_items": 0.0}
    return {"hit": [100 * h / n for h in hits], "mrr": 100 * rr / n, "n_queries": n, "mean_items": n_items / n}


def rank(runs: List[RunVotes], k_max: int, seed: int, embedding_model: str) -> Dict[str, dict]:
    """{setting: {method: result}} on the documents every run covers."""
    common = set.intersection(*(set(r.docs) for r in runs))
    by_id = runs[0].eval_docs()
    docs = [by_id[i] for i in sorted(common) if i in by_id]
    votes = [(f"votes {r.name.rsplit('_', 1)[-1]} n={r.n_runs}", edge_votes(r)) for r in runs]
    tf, emb = tfidf(docs), embedding(docs, embedding_model)
    out = {}
    for directed in (True, False):
        rows = {}
        for label, v in votes:
            fn = ((lambda d, q, c, v=v: v[d.id][(c, q)]) if directed
                  else (lambda d, q, c, v=v: v[d.id][(c, q)] + v[d.id][(q, c)]))
            rows[label] = evaluate(docs, fn, directed, k_max, seed)
        rows["tfidf cosine (trigger: sentence)"] = evaluate(docs, tf, directed, k_max, seed)
        rows[f"embedding cosine ({embedding_model})"] = evaluate(docs, emb, directed, k_max, seed)
        rows["random"] = evaluate(docs, lambda d, q, c: 0.0, directed, k_max, seed)
        out["directed" if directed else "undirected"] = {"n_docs": len(docs), "methods": rows}
    return out
