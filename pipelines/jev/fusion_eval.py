"""Frozen-rule fusion of a document-level vote with d1-ultra, per corpus; zero API calls.

Rule fixed in R3 (no refit): score = (vote(a,b) + vote(b,a) + d1(a,b) + d1(b,a)) / 4, positive at >= 0.39.
Binary existence on the evaluated ordered candidates; gold read from each run's frozen suite.json.
"""
import ast
import json
import random
from collections import defaultdict
from pathlib import Path

from benchmark.labels import is_norel

THRESHOLD = .39


def _lit(value):
    return value if isinstance(value, list) else ast.literal_eval(value) if value else []


def _load(run_dir):
    run_dir = Path(run_dir)
    suite = json.loads((run_dir / "suite.json").read_text())
    gold = {}
    for doc in suite["documents"]:
        for a, label, b in _lit(doc["gold"]):
            if not is_norel(label):
                gold[doc["id"], str(a), str(b)] = True
    scores, model = {}, None
    for row in map(json.loads, (run_dir / "predictions.jsonl").read_text().splitlines()):
        for s in row["metadata"]["decision_scores"]:
            scores[row["id"], str(s["source"]), str(s["target"])] = float(s["positive_probability"])
        model = row["metadata"].get("model", model)
    metadata = json.loads((run_dir / "metadata.json").read_text())
    return scores, gold, metadata


def _counts(pred, gold, keys):
    tp = sum(1 for k in keys if pred[k] and gold.get(k, False))
    fp = sum(1 for k in keys if pred[k] and not gold.get(k, False))
    fn = sum(1 for k in keys if not pred[k] and gold.get(k, False))
    return tp, fp, fn


def _f1(c):
    tp, fp, fn = c
    return 200 * tp / (2 * tp + fp + fn) if tp else 0.


def summary(pred, gold, keys):
    tp, fp, fn = _counts(pred, gold, keys)
    return {"P": round(100 * tp / max(tp + fp, 1), 2), "R": round(100 * tp / max(tp + fn, 1), 2),
            "F1": round(_f1((tp, fp, fn)), 2), "tp": tp, "fp": fp, "fn": fn}


def paired(a, b, gold, keys, samples=2000, seed=42):
    by_doc = defaultdict(list)
    for k in keys:
        by_doc[k[0]].append(k)
    docs = sorted(by_doc)
    per = {d: (_counts(a, gold, by_doc[d]), _counts(b, gold, by_doc[d])) for d in docs}
    rng, values = random.Random(seed), []
    for _ in range(samples):
        draw = [rng.choice(docs) for _ in docs]
        values.append(_f1([sum(per[d][0][i] for d in draw) for i in range(3)])
                      - _f1([sum(per[d][1][i] for d in draw) for i in range(3)]))
    values.sort()
    return {"delta": round(_f1(_counts(a, gold, keys)) - _f1(_counts(b, gold, keys)), 2),
            "ci95": [round(values[int(.025 * samples)], 2), round(values[int(.975 * samples) - 1], 2)],
            "documents": len(docs)}


def evaluate(receipt, corpora):
    """`corpora`: {name: {"d1": run-name prefix, "votes": run-name prefix, "historical": {...}}}."""
    runs = [r for r in receipt["runs"] if r["status"] in {"completed", "completed_with_errors"}]
    report = {"rule": "mean of symmetrised vote and d1 >= 0.39 (frozen in R3, no refit)", "corpora": {}}
    for name, spec in corpora.items():
        d1_runs = [r for r in runs if Path(r["run_dir"]).name.startswith(spec["d1"])]
        vote_runs = [r for r in runs if Path(r["run_dir"]).name.startswith(spec["votes"])]
        expected = spec.get("documents")
        if len(d1_runs) != 1:
            raise ValueError(f"{name}: expected one completed d1 run, found {len(d1_runs)}")
        d1, gold, _ = _load(d1_runs[0]["run_dir"])
        votes, models = {}, set()
        for r in vote_runs:
            s, _, meta = _load(r["run_dir"])
            votes.update(s)
            models.update(meta.get("served_models", []))
        docs = {k[0] for k in d1}
        if expected is not None and (len(docs) != expected or len({k[0] for k in votes}) != expected):
            raise ValueError(f"{name}: incomplete coverage (d1 {len(docs)} docs, votes "
                             f"{len({k[0] for k in votes})} docs, expected {expected}); no metric reported")
        if set(votes) != set(d1) or not all((k[0], k[2], k[1]) in d1 for k in d1):
            raise ValueError(f"{name}: vote/d1 candidate mismatch")
        keys = sorted(d1)
        sym = lambda s, k: (s[k] + s[k[0], k[2], k[1]]) / 2
        fusion = {k: (sym(votes, k) + sym(d1, k)) / 2 >= THRESHOLD for k in keys}
        d1_raw = {k: d1[k] >= .5 for k in keys}
        vote_major = {k: votes[k] >= 4 / 7 - 1e-9 for k in keys}
        report["corpora"][name] = {
            "documents": len(docs), "ordered_candidates": len(keys), "gold_positive": sum(gold.get(k, False) for k in keys),
            "served_vote_models": sorted(models),
            "fusion": summary(fusion, gold, keys), "d1_alone_threshold_0.5": summary(d1_raw, gold, keys),
            "vote_alone_majority_4_of_7": summary(vote_major, gold, keys),
            "paired_fusion_minus_d1": paired(fusion, d1_raw, gold, keys),
            "paired_fusion_minus_vote": paired(fusion, vote_major, gold, keys),
            "historical_unpaired_reference": spec.get("historical")}
    return report
