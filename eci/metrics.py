"""Set, multiclass and binary metrics. Arithmetic identical to agentere's utils/metrics.py, so
every number a workere run logs is comparable with the agentere history."""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

from eci.labels import is_norel


def prf(tp: int, fp: int, fn: int) -> tuple:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def set_metrics(gold: Iterable[tuple], pred: Iterable[tuple]) -> Dict[str, float]:
    g, p = set(gold), set(pred)
    tp = len(g & p)
    prec, rec, f1 = prf(tp, len(p - g), len(g - p))
    return {"precision": prec, "recall": rec, "f1": f1, "support": len(g), "predicted": len(p), "tp": tp}


def multiclass(y_true: Sequence[str], y_pred: Sequence[str], labels: Iterable[str]) -> dict:
    """Per-label P/R/F1, macro F1 and micro P/R/F1 with every norel variant excluded from both
    averages. Labels outside `labels` that occur in the data are tracked too."""
    y_true = [str(x) for x in y_true]
    y_pred = [str(x) for x in y_pred]
    unique = set(labels) | set(y_true) | set(y_pred)
    counts = {lab: [0, 0, 0] for lab in unique}  # tp, fp, fn
    for gt, pr in zip(y_true, y_pred):
        if gt == pr:
            counts[gt][0] += 1
        else:
            counts[gt][2] += 1
            counts[pr][1] += 1
    per_label = {}
    for lab in unique:
        tp, fp, fn = counts[lab]
        p, r, f = prf(tp, fp, fn)
        per_label[lab] = {"precision": p, "recall": r, "f1": f, "support": tp + fn}
    positive = [lab for lab in unique if not is_norel(lab)]
    macro = sum(per_label[l]["f1"] for l in positive) / len(positive) if positive else 0.0
    tp = sum(counts[l][0] for l in positive)
    fp = sum(counts[l][1] for l in positive)
    fn = sum(counts[l][2] for l in positive)
    p, r, f = prf(tp, fp, fn)
    return {"per_label": per_label, "macro_f1": macro, "micro_precision": p,
            "micro_recall": r, "micro_f1": f, "total": len(y_true)}


def binary(y_true: Sequence[str], y_pred: Sequence[str]) -> dict:
    """Any relation positive, norel negative."""
    tp = fp = fn = pos = 0
    for gt, pr in zip(y_true, y_pred):
        g, p = not is_norel(gt), not is_norel(pr)
        pos += g
        if g and p:
            tp += 1
        elif p:
            fp += 1
        elif g:
            fn += 1
    p, r, f = prf(tp, fp, fn)
    return {"precision": p, "recall": r, "f1": f, "support_pos": pos}


def split_by_relation(y_true: List[str], y_pred: List[str], rels: List[str]):
    """(intra_true, intra_pred, inter_true, inter_pred, n_unknown). Pairs whose sentence cannot
    be resolved are excluded from BOTH subsets, never folded into inter."""
    it, ip, et, ep, unknown = [], [], [], [], 0
    for gt, pr, rel in zip(y_true, y_pred, rels):
        if rel == "intra":
            it.append(gt)
            ip.append(pr)
        elif rel == "inter":
            et.append(gt)
            ep.append(pr)
        else:
            unknown += 1
    return it, ip, et, ep, unknown
