"""Score model outputs without importing inference, tools, prompts or tracking."""
from __future__ import annotations

import random

from benchmark import report, voting
from benchmark.data import label_set, to_binary_undirected
from benchmark.io import fingerprint
from benchmark.labels import BINARY_LABELS, NOREL
from benchmark.metrics import set_metrics
from benchmark.suite import documents


def _align(docs, predictions):
    by_id = {}
    for row in predictions:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise ValueError("every prediction needs a string document id")
        if row["id"] in by_id:
            raise ValueError(f"duplicate prediction: {row['id']}")
        by_id[row["id"]] = row
    expected = {doc.id for doc in docs}
    if set(by_id) != expected:
        raise ValueError(f"document coverage mismatch: missing={sorted(expected-set(by_id))[:5]}, "
                         f"extra={sorted(set(by_id)-expected)[:5]}")
    return by_id


def _triples(value):
    if not isinstance(value, list):
        raise ValueError("triples must be an array of [source, label, target]")
    for triple in value:
        if (not isinstance(triple, (list, tuple)) or len(triple) != 3
                or any(not isinstance(x, str) or not x for x in triple)
                or "," in triple[0] or "," in triple[2]):
            raise ValueError(f"invalid relation triple: {triple!r}")
    # Deliberately no normalization: historical off-spec labels remain measurable.
    return [tuple(triple) for triple in value]


def evaluate(suite, predictions, *, model="external", representation="final"):
    """Final triples OR preprocessed passes; the representation is fixed for the whole run.

    Missing documents are errors. An explicit empty list is a real negative prediction.
    Candidate constraints are an inference instruction historically, not a scoring filter.
    Unknown mention IDs and off-spec labels are consequently scored, never silently dropped.
    """
    docs = documents(suite)
    predictions = list(predictions)
    by_id = _align(docs, predictions)
    if representation not in ("final", "passes"):
        raise ValueError("representation must be final or passes")
    cfg, results = suite["config"], []
    data, rs = cfg["data"], cfg["experiment"]["resampling"]
    n = rs["n_runs"] if rs["enabled"] else 1
    for doc in docs:
        row = by_id[doc.id]
        allowed = {"id", "triples"} if representation == "final" else {"id", "passes"}
        if set(row) != allowed:
            raise ValueError(f"{doc.id}: expected fields {sorted(allowed)}")
        if representation == "passes":
            if not isinstance(row["passes"], list) or len(row["passes"]) != n:
                raise ValueError(f"{doc.id}: expected {n} preprocessed passes")
            passes = [_triples(p) for p in row["passes"]]
            if n > 1:
                final, stats = voting.aggregate(passes, rs["tie_breaking"],
                                               random.Random(f"{cfg['seed']}::{doc.id}::vote"))
            else:
                final, stats = passes[0], voting.single_pass_stats(passes[0])
            if data["binary_undirected"]:
                final = [(min(s, t), lab, max(s, t)) for s, lab, t in final]
            if data["vote_to_binary"]:
                final = voting.votes_to_binary(stats, data["binary_default"], data["novote_norel"])
        else:
            final = _triples(row["triples"])
            stats = voting.single_pass_stats(final)
        gold = doc.gold
        if data["vote_to_binary"]:
            gold = [t for t in to_binary_undirected(gold) if t[1] != NOREL]
        results.append({"id": doc.id, "doc_idx": doc.idx, "lang": doc.lang,
                        "gold_triples": gold, "pred_triples": final, "pair_stats": stats,
                        "mention_sentence": doc.sentence_of, "first_token": doc.first_token,
                        "surfaces": doc.surfaces, "justifications": {}, "failure": None})
    labels = BINARY_LABELS if data["binary_undirected"] or data["vote_to_binary"] else label_set(cfg)
    scores = report.build(results, len(docs), cfg, labels)
    kf = cfg["datasets"][cfg["active_dataset"]].get("kfold", {})
    if kf.get("enabled"):
        scores["kfold_n_folds"] = kf["n_folds"]
    return {"schema": "workere-benchmark-result/v1", "task": "relations",
            "suite_fingerprint": suite["fingerprint"], "protocol": suite["protocol"],
            "representation": representation, "model": model,
            "predictions_fingerprint": fingerprint(predictions), "config": cfg, "results": scores}


def _spans(spans, n_tokens):
    if not isinstance(spans, list):
        raise ValueError("spans must be an array")
    out = []
    for span in spans:
        if (not isinstance(span, list) or not span
                or any(type(t) is not int or not 0 <= t < n_tokens for t in span)
                or span != sorted(set(span))):
            raise ValueError(f"invalid token-index span: {span!r}")
        out.append(tuple(span))
    return set(out)


def evaluate_event_mentions(suite, predictions, *, model="external"):
    """NEW opt-in task: exact, untyped event spans; never an historical relation score."""
    docs = documents(suite)
    predictions = list(predictions)
    by_id = _align(docs, predictions)
    gold, pred, per_doc = [], [], []
    for doc in docs:
        if len(doc.mentions) != len(doc.spans) or not doc.tokens:
            raise ValueError(f"{doc.id}: no usable event mention annotations")
        row = by_id[doc.id]
        if set(row) != {"id", "spans"}:
            raise ValueError("event predictions require exactly id and spans")
        g = _spans(doc.spans, len(doc.tokens))
        p = _spans(row["spans"], len(doc.tokens))
        gold.extend((doc.id, s) for s in g)
        pred.extend((doc.id, s) for s in p)
        per_doc.append({"id": doc.id, **set_metrics(g, p)})
    return {"schema": "workere-benchmark-result/v1", "task": "event_mentions",
            "protocol": "event-mentions-exact-untyped/v1", "representation": "spans",
            "suite_fingerprint": suite["fingerprint"], "model": model,
            "predictions_fingerprint": fingerprint(predictions),
            "results": {**set_metrics(gold, pred), "per_doc_metrics": per_doc}}


def compare(reports):
    """Reject different datasets/protocols/representations instead of ranking unlike scores."""
    if not reports:
        raise ValueError("provide at least one result")
    fields = ("suite_fingerprint", "protocol", "task", "representation")
    identity = tuple(reports[0][k] for k in fields)
    if any(tuple(r[k] for k in fields) != identity for r in reports):
        raise ValueError("incomparable results: suite, task, protocol or representation differ")
    task = reports[0]["task"]
    return [{"model": r["model"], **({k: r["results"][k] for k in ("micro_f1", "macro_f1", "binary")}
                                    if task == "relations" else
                                    {k: r["results"][k] for k in ("precision", "recall", "f1")})}
            for r in reports]
