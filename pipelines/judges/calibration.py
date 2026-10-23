"""Fit and freeze score-combination rules on DEV44 only."""
import json
from pathlib import Path
import numpy as np

from benchmark.io import fingerprint
from pipelines.judges.common import GRID, Dev44, Scorer, f1_arr


def training_data(controls, suites, judge_result):
    data = Dev44(controls, suites)
    source = json.loads(Path(judge_result).read_text())
    judge = {d: {tuple(k.split(",")): p for k, p in m.items()}
             for d, m in source["judge_probabilities"].items()}
    if set(judge) != set(data.docs):
        raise ValueError("training judge population differs")
    return data, judge


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -50, 50)))


def features(fused, judge):
    values = np.column_stack((fused, judge))
    values = np.clip(values, 1e-4, 1 - 1e-4)
    return np.log(values / (1 - values))


def fit_logistic(data, judge, docs, penalty=1.0):
    """Three parameters, L2 penalty on two standardized logit-score coefficients."""
    rows = [(data.fused[d][k], p, bool(data.by_doc[d][k]["gold"]))
            for d in docs for k, p in judge[d].items()]
    x = features([r[0] for r in rows], [r[1] for r in rows])
    y = np.array([r[2] for r in rows], float)
    mean, scale = x.mean(0), np.maximum(x.std(0), 1e-6)
    x = np.column_stack((np.ones(len(x)), (x - mean) / scale))
    reg = np.array([0., penalty, penalty])
    beta = np.array([np.log((y.sum() + .5) / (len(y) - y.sum() + .5)), 0., 0.])

    def objective(b):
        z = x @ b
        return float(np.logaddexp(0, z).sum() - y @ z + .5 * (reg * b * b).sum())

    for _ in range(100):
        p = sigmoid(x @ beta)
        gradient = x.T @ (p - y) + reg * beta
        hessian = (x.T * (p * (1 - p))) @ x + np.diag(reg + 1e-9)
        step = np.linalg.solve(hessian, gradient)
        size = 1.
        while size > 1e-8 and objective(beta - size * step) > objective(beta):
            size *= .5
        beta -= size * step
        if np.max(np.abs(size * step)) < 1e-8:
            break
    return {"kind": "logistic", "mean": mean.tolist(), "scale": scale.tolist(),
            "coefficients": beta.tolist(), "l2_penalty": penalty}


def logistic_prob(rule, f, j):
    x = (features(f, j) - np.array(rule["mean"])) / np.array(rule["scale"])
    return sigmoid(np.column_stack((np.ones(len(x)), x)) @ np.array(rule["coefficients"]))


def rule_scores(data, judge, rule, doc):
    if rule["kind"] == "mean":
        w = rule["weight"]
        return {k: f if tuple(sorted(k)) not in judge[doc]
                else (1 - w) * f + w * judge[doc][tuple(sorted(k))]
                for k, f in data.fused[doc].items()}
    out = {k: float(f >= .39) for k, f in data.fused[doc].items()}
    keys = list(judge[doc])
    probs = logistic_prob(rule, [data.fused[doc][k] for k in keys], [judge[doc][k] for k in keys])
    for (a, b), p in zip(keys, probs):
        for key in ((a, b), (b, a)):
            if key in out:
                out[key] = float(p)
    return out


def best_threshold(data, judge, rule, docs):
    scorer = Scorer(data)
    counts = sum(scorer.grid(d, rule_scores(data, judge, rule, d)) for d in docs)
    i = int(np.argmax(f1_arr(counts)))
    return float(GRID[i]), float(f1_arr(counts)[i])


def fit_rules(data, judge):
    best = None
    for w in (0, .25, .5, .75, 1):
        rule = {"kind": "mean", "weight": w}
        threshold, f1 = best_threshold(data, judge, rule, data.docs)
        if best is None or f1 > best[0] + 1e-12:
            best = (f1, {**rule, "threshold": threshold})
    logistic = fit_logistic(data, judge, data.docs)
    threshold, f1 = best_threshold(data, judge, logistic, data.docs)
    return {"mean": {**best[1], "training_f1": best[0]},
            "logistic": {**logistic, "threshold": threshold, "training_f1": f1}}


def freeze(controls, suites, judge_result):
    data, judge = training_data(controls, suites, judge_result)
    rules = fit_rules(data, judge)
    bundle = {"schema": "workere-calibration-freeze/v1", "training_documents": data.docs,
              "training_controls_hash": fingerprint(json.loads(Path(controls).read_text())),
              "training_judge_hash": fingerprint(json.loads(Path(judge_result).read_text())["judge_probabilities"]),
              "rules": rules, "fit_labels": "DEV44 only; target reserve labels never passed to fitter",
              "selection": "maximum training micro F1; no performance claim from training values",
              "source_j2_run": "582ddbdd-9f2b-4878-bd7d-650e2874869b"}
    bundle["fingerprint"] = fingerprint(bundle)
    return bundle


def load_freeze(path):
    bundle = json.loads(Path(path).read_text())
    if fingerprint({k: v for k, v in bundle.items() if k != "fingerprint"}) != bundle["fingerprint"]:
        raise ValueError("frozen calibration fingerprint differs")
    return bundle
