"""Document results -> the run report.

The report keeps agentere's schema key for key (`per_pair_predictions`, `binary_intra`,
`per_doc_metrics`, ...), so the agentere analysis scripts that read `run_*.json` work on workere
runs unchanged. Additions are purely additive:

  per-pair rows   `word_distance`, `sentence_distance` — so distance histograms no longer need
                  to re-download the dataset to recover token positions
  doc_index       per document: mention -> sentence, mention -> first token, mention -> surface.
                  Post-hoc transforms that create pairs no row mentions (coreference
                  propagation, transitive closure) need the sentence of every mention to put a
                  new pair in intra or inter; agentere had to reload HuggingFace for that.
  failures        one entry per document that failed after its retries, with the error
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List

from benchmark import metrics as M
from benchmark.labels import NOREL


def pair_rows(res: Dict[str, Any]) -> Dict[str, Any]:
    """Every pair the gold, the prediction or any vote mentions, as report rows.

    Voted-but-rejected pairs (causal in some passes, outvoted to norel) are included: without
    them a vote-threshold analysis would see the rejected true positives (gold rows) and none of
    the rejected false positives.
    """
    gold = {(s, t): l for s, l, t in res["gold_triples"]}
    pred = {(s, t): l for s, l, t in res["pred_triples"]}
    stats = res["pair_stats"]
    sent = res["mention_sentence"]
    first = res["first_token"]
    just = res["justifications"]
    pairs = set(gold) | set(pred)
    pairs |= {tuple(k.split(",")) for k in stats if len(k.split(",")) == 2}

    rows, y_true, y_pred, rels = [], [], [], []
    for src, tgt in sorted(pairs):
        g, p = gold.get((src, tgt), NOREL), pred.get((src, tgt), NOREL)
        sa, sb = sent.get(src), sent.get(tgt)
        rel = "unknown" if sa is None or sb is None else ("intra" if sa == sb else "inter")
        fa, fb = first.get(src), first.get(tgt)
        key = f"{src},{tgt}"
        rows.append({
            "doc_idx": res["doc_idx"], "id": res["id"], "lang": res["lang"], "pair": key,
            "gold": g, "pred": p,
            "vote_counts": stats.get(key, {}).get("vote_counts"),
            "sentence_relation": rel,
            "justifications": just.get(key) or None,
            "word_distance": abs(fa - fb) if fa is not None and fb is not None else None,
            "sentence_distance": abs(sa - sb) if sa is not None and sb is not None else None,
        })
        y_true.append(g)
        y_pred.append(p)
        rels.append(rel)
    return {"rows": rows, "y_true": y_true, "y_pred": y_pred, "rels": rels}


def _scope_metrics(y_true, y_pred, rels, labels) -> dict:
    it, ip, et, ep, unknown = M.split_by_relation(y_true, y_pred, rels)
    return {
        "multiclass": M.multiclass(y_true, y_pred, labels),
        "binary": M.binary(y_true, y_pred),
        "binary_intra": M.binary(it, ip),
        "binary_inter": M.binary(et, ep),
        "sentence_unknown_pairs": unknown,
        "total_pairs": len(y_true),
    }


def build(results: List[Dict[str, Any]], total_docs: int, cfg: Dict[str, Any],
          labels: Iterable[str]) -> Dict[str, Any]:
    labels = list(labels)
    all_true, all_pred, all_rels = [], [], []
    per_doc, per_pair = [], []
    per_lang = defaultdict(lambda: ([], [], []))
    for res in results:
        al = pair_rows(res)
        all_true += al["y_true"]
        all_pred += al["y_pred"]
        all_rels += al["rels"]
        per_pair += al["rows"]
        lt, lp, lr = per_lang[res["lang"]]
        lt += al["y_true"]
        lp += al["y_pred"]
        lr += al["rels"]
        d = _scope_metrics(al["y_true"], al["y_pred"], al["rels"], labels)
        per_doc.append({
            "doc_idx": res["doc_idx"], "id": res["id"], "pairs": d["total_pairs"],
            "macro_f1": d["multiclass"]["macro_f1"], "micro_f1": d["multiclass"]["micro_f1"],
            "micro_precision": d["multiclass"]["micro_precision"],
            "micro_recall": d["multiclass"]["micro_recall"],
            "per_label": d["multiclass"]["per_label"],
            "binary": d["binary"], "binary_intra": d["binary_intra"],
            "binary_inter": d["binary_inter"],
            "sentence_unknown_pairs": d["sentence_unknown_pairs"],
        })

    g = _scope_metrics(all_true, all_pred, all_rels, labels)
    per_lang_metrics = {lang: _scope_metrics(t, p, r, labels)
                        for lang, (t, p, r) in per_lang.items() if t}
    total = g["total_pairs"]
    rs = cfg["experiment"]["resampling"]
    return {
        "per_label": g["multiclass"]["per_label"],
        "macro_f1": g["multiclass"]["macro_f1"],
        "micro_precision": g["multiclass"]["micro_precision"],
        "micro_recall": g["multiclass"]["micro_recall"],
        "micro_f1": g["multiclass"]["micro_f1"],
        "total_pairs": g["multiclass"]["total"],
        "binary": g["binary"],
        "binary_intra": g["binary_intra"],
        "binary_inter": g["binary_inter"],
        "sentence_unknown_pairs": g["sentence_unknown_pairs"],
        "sentence_unknown_pct": g["sentence_unknown_pairs"] / total if total else 0.0,
        "skipped_docs": total_docs - len(results),
        "retry_failed_docs": sum(1 for r in results if r["failure"]),
        "failures": [{"id": r["id"], "error": r["failure"]} for r in results if r["failure"]],
        "per_doc_metrics": per_doc,
        "per_lang_metrics": per_lang_metrics,
        "per_pair_predictions": per_pair,
        "doc_index": {r["id"]: {"mention_sentence": r["mention_sentence"],
                                "first_token": r["first_token"], "surfaces": r["surfaces"]}
                      for r in results},
        "resampling": {"enabled": rs["enabled"], "n_runs": rs["n_runs"],
                       "tie_breaking": rs["tie_breaking"]},
    }


def headline(report: Dict[str, Any]) -> str:
    """The lines printed at the end of a run."""
    lines = []
    for scope in ("binary", "binary_intra", "binary_inter"):
        m = report[scope]
        lines.append(f"{scope:13s} P {m['precision']:.4f}  R {m['recall']:.4f}  "
                     f"F1 {m['f1']:.4f}  gold+ {m['support_pos']}")
    lines.append(f"micro         P {report['micro_precision']:.4f}  R {report['micro_recall']:.4f}  "
                 f"F1 {report['micro_f1']:.4f}  macro F1 {report['macro_f1']:.4f}  "
                 f"pairs {report['total_pairs']}")
    if report["sentence_unknown_pct"] > 0.01:
        lines.append(f"WARNING: {report['sentence_unknown_pairs']} pairs "
                     f"({report['sentence_unknown_pct']:.1%}) have unresolved sentences and are "
                     f"excluded from both binary_intra and binary_inter")
    if report["retry_failed_docs"]:
        lines.append(f"WARNING: {report['retry_failed_docs']} document(s) failed after all "
                     f"retries and are scored as empty predictions — resume the run to retry them")
    if len(report["per_lang_metrics"]) > 1:
        for lang, lm in sorted(report["per_lang_metrics"].items()):
            b, mc = lm["binary"], lm["multiclass"]
            lines.append(f"  [{lang}] pairs={lm['total_pairs']} micro_f1={mc['micro_f1']:.4f} "
                         f"binary f1={b['f1']:.4f} p={b['precision']:.4f} r={b['recall']:.4f}")
    return "\n".join(lines)
