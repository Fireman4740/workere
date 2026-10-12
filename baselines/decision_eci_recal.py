"""Recalibrate saved Laya outputs (decision_eci.py jsonl) and re-score them.

The model card and its companion write-ups say Laya ships over-confident and that one temperature
per (question type, option count) must be refit on target data. The saved probabilities were
produced with the shipped temperature T_s, so logits are recovered as T_s * log p (up to a
per-row constant, which softmax ignores) and refit here:

    shipped        as run (T_s from rl_agent_config.json)
    temperature    one T, fitted by NLL — the card's recipe
    bias+temp      one T plus a bias per option (vector scaling): also removes the option-A lean

Every fit is 2-fold cross-fitted over documents (fit on one half, applied to the other), so the
calibration numbers are out-of-sample.

    decision_eci_recal.py <name>:<jsonl>:<T_shipped> ...
"""
from __future__ import annotations

import json
import sys
import zlib

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from decision_eci import VARIANTS  # noqa: E402


def load(path, t_ship):
    rows = [json.loads(l) for l in open(path)]
    v = rows[0]["variant"]
    keys = list(VARIANTS[v][0]["criteria"])
    none_opt, fwd, bwd = VARIANTS[v][1:4]
    p = np.array([[max(r["probs"][k], 1e-4) for k in keys] for r in rows])
    z = t_ship * np.log(p)
    if fwd:   # dir3: the gold option is the direction
        tgt = [keys.index(fwd if r["gold_dir"] == "12" else bwd if r["gold_dir"] == "21" else none_opt)
               for r in rows]
    else:
        yes = [k for k in keys if k != none_opt][0]
        tgt = [keys.index(yes if r["gold"] else none_opt) for r in rows]
    fold = np.array([zlib.crc32(r["doc"].encode()) % 2 for r in rows])
    return z, np.array(tgt), np.array([r["gold"] for r in rows]), keys.index(none_opt), fold


def fit(z, tgt, bias: bool):
    zt, yt = torch.tensor(z, dtype=torch.float64), torch.tensor(tgt)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(z.shape[1], dtype=torch.float64, requires_grad=bias)
    opt = torch.optim.LBFGS([log_t, b] if bias else [log_t], max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(zt / log_t.exp() + b, yt)
        loss.backward()
        return loss
    opt.step(closure)
    return float(log_t.exp()), b.detach().numpy()


def softmax(x):
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def ece(p, tgt, bins=15):
    conf, correct = p.max(1), (p.argmax(1) == tgt)
    edges = np.linspace(0, 1, bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            out += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return out


def f1(y, pred):
    tp = int((y & pred).sum()); fp = int((~y & pred).sum()); fn = int((y & ~pred).sum())
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def oracle_f1(y, s):
    from sklearn.metrics import precision_recall_curve
    pr, rc, _ = precision_recall_curve(y, s)
    return float(np.max(np.where(pr + rc > 0, 2 * pr * rc / np.maximum(pr + rc, 1e-12), 0)))


def metrics(p, tgt, y, none_idx):
    s = 1 - p[:, none_idx]
    yb = y.astype(bool)
    return {"NLL": float(-np.log(p[np.arange(len(tgt)), tgt]).mean()), "ECE": ece(p, tgt),
            "Brier": float(((s - y) ** 2).mean()), "mean P(causal)": float(s.mean()),
            "AUROC": roc_auc_score(y, s), "AP": average_precision_score(y, s),
            "argmax F1": f1(yb, p.argmax(1) != none_idx), "F1 @ P>=0.5": f1(yb, s >= 0.5),
            "oracle F1": oracle_f1(y, s)}


def main():
    cols = ["NLL", "ECE", "Brier", "mean P(causal)", "AUROC", "AP", "argmax F1", "F1 @ P>=0.5", "oracle F1"]
    print("| run | method | fitted T (fold 0 / 1) | " + " | ".join(cols) + " |")
    print("|---|---|---|" + "---|" * len(cols))
    for spec in sys.argv[1:]:
        name, path, t_ship = spec.rsplit(":", 2)
        z, tgt, y, none_idx, fold = load(path, float(t_ship))
        base = z / float(t_ship)
        results = {"shipped": (softmax(base), f"{float(t_ship):.2f} (shipped)")}
        for method, bias in (("temperature", False), ("bias+temp", True)):
            logits, ts = np.zeros_like(z), []
            for k in (0, 1):
                t, b = fit(z[fold != k], tgt[fold != k], bias)
                logits[fold == k] = z[fold == k] / t + b
                ts.append(f"{t:.2f}")
            results[method] = (softmax(logits), " / ".join(ts))
        for method, (p, tdesc) in results.items():
            m = metrics(p, tgt, y, none_idx)
            print(f"| {name} | {method} | {tdesc} | " + " | ".join(f"{m[c]:.3f}" for c in cols) + " |")


if __name__ == "__main__":
    main()
