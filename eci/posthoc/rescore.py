"""Re-score a finished run under a different decision rule, or recompute its metrics.

  recompute       every metric of the report rebuilt from its per-pair rows — checks a report
                  against the current metric code (agentere: recompute_metrics.py)
  vote_to_binary  as if data.vote_to_binary had been on: directed votes of (a,b) and (b,a)
                  summed causal-against-norel per unordered pair, for all four
                  binary_default x novote_norel rules — calls the pipeline's own function, so
                  it cannot drift from what a live run reports (agentere: vote_to_binary_eval.py)
  or / and        direction-agnostic binary: an unordered pair is causal if either direction
                  (OR) or both directions (AND; single-direction pairs as they are) were
                  predicted causal (agentere: binary_eval.py)
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from eci import metrics as M
from eci.labels import BINARY_LABELS, BINARY_POS, CAUSAL, NOREL, is_norel
from eci.posthoc.runs import RunVotes
from eci.voting import votes_to_binary


def recompute(run: RunVotes) -> Dict[str, dict]:
    """{"logged": ..., "recomputed": ...} for the headline metrics."""
    rows = run.results["per_pair_predictions"]
    labels = BINARY_LABELS if run.cfg["data"]["binary_undirected"] else list(run.cfg["data"]["labels"])
    yt = [r["gold"] for r in rows]
    yp = [r["pred"] for r in rows]
    rel = [r.get("sentence_relation", "unknown") for r in rows]
    it, ip, et, ep, _ = M.split_by_relation(yt, yp, rel)
    mc = M.multiclass(yt, yp, labels)
    new = {"micro_f1": mc["micro_f1"], "macro_f1": mc["macro_f1"], "binary": M.binary(yt, yp),
           "binary_intra": M.binary(it, ip), "binary_inter": M.binary(et, ep)}
    old = {k: run.results[k] for k in new}
    return {"logged": old, "recomputed": new}


def _undirected_view(run: RunVotes):
    """doc -> unordered pair -> (gold causal?, relation, {direction key: vote_counts})."""
    out = defaultdict(dict)
    for row in run.results["per_pair_predictions"]:
        a, b = (x.strip() for x in row["pair"].split(",", 1))
        if a == b:
            continue
        key = frozenset((a, b))
        g, rel, dirs = out[row["id"]].get(key, (False, row.get("sentence_relation", "unknown"), {}))
        dirs[f"{a},{b}"] = {"vote_counts": row.get("vote_counts") or {row["pred"]: run.n_runs},
                            "pred": row["pred"]}
        out[row["id"]][key] = (g or row["gold"] in CAUSAL, rel, dirs)
    return out


def _prf(decisions: List[tuple]) -> Dict[str, tuple]:
    agg = {s: [0, 0, 0] for s in ("all", "intra", "inter")}
    for gold, pred, rel in decisions:
        for s in ("all", rel):
            if s in agg and (gold or pred):
                agg[s][0 if gold and pred else 1 if pred else 2] += 1
    return {s: tuple(100 * x for x in M.prf(*v)) for s, v in agg.items()}


def binary_rules(run: RunVotes) -> Dict[str, Dict[str, tuple]]:
    """{rule name: {scope: (P, R, F1)}} on unordered pairs."""
    view = _undirected_view(run)
    rules = {}
    for default in ("norel", "causal"):
        for novote in (True, False):
            dec = []
            for doc, pairs in view.items():
                stats = {k: v for _, _, dirs in pairs.values() for k, v in dirs.items()}
                pos = {frozenset((s, t)) for s, _, t in votes_to_binary(stats, default, novote)}
                dec += [(g, key in pos, rel) for key, (g, rel, _) in pairs.items()]
            rules[f"vote_to_binary default={default} novote_norel={novote}"] = _prf(dec)
    for name, need_both in (("OR (either direction)", False), ("AND (both directions)", True)):
        dec = []
        for doc, pairs in view.items():
            for key, (g, rel, dirs) in pairs.items():
                causal = [not is_norel(d["pred"]) for d in dirs.values()]
                pred = all(causal) if need_both else any(causal)
                dec.append((g, pred, rel))
        rules[name] = _prf(dec)
    return rules
