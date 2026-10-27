"""Gold-separated evaluation for projected positive predictions."""
from __future__ import annotations
from collections import defaultdict
import math
import random
from typing import Any
from .core import unordered


def parse_pair_list(values: list[list[str]]) -> set[tuple[str, str]]:
    result = set()
    for pair in values:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError("pair must be length-2 list")
        a, b = pair
        if not isinstance(a, str) or not isinstance(b, str) or a == b:
            raise ValueError("invalid unordered pair")
        result.add(unordered(a, b))
    return result


def measures(tp: int, pred: int, gold: int) -> dict[str, float | int]:
    fp, fn = pred-tp, gold-tp
    if min(tp,fp,fn) < 0:
        raise ValueError("negative confusion count")
    return {
        "TP":tp, "FP":fp, "FN":fn,
        "precision":tp/pred if pred else 0.0,
        "recall":tp/gold if gold else 0.0,
        "F1":2*tp/(pred+gold) if pred+gold else 0.0,
        "predicted": pred, "gold":gold,
    }


def accumulate(rows: list[tuple[int,int,int]]) -> dict[str, float | int]:
    return measures(sum(x[0] for x in rows), sum(x[1] for x in rows), sum(x[2] for x in rows))


def bootstrap_diff_by_topic(
    per_topic: dict[str, dict[str, tuple[int,int,int]]], a: str, b: str,
    replicates: int=2000, seed: int=27
) -> dict[str, Any]:
    topics = sorted(per_topic)
    if len(topics) < 3:
        return {"ci95": None, "reason":"fewer than 3 topics", "topics":len(topics)}
    rng = random.Random(seed)
    diffs = []
    for _ in range(replicates):
        sample = [topics[rng.randrange(len(topics))] for _ in topics]
        fa = accumulate([per_topic[t][a] for t in sample])["F1"]
        fb = accumulate([per_topic[t][b] for t in sample])["F1"]
        diffs.append(float(fa)-float(fb))
    diffs.sort()
    return {"ci95":[diffs[int((replicates-1)*0.025)], diffs[int((replicates-1)*0.975)]],
            "topics":len(topics), "replicates":replicates,
            "unit":"topic", "caveat":"exploratory, may be unstable with few topics"}


def evaluate(
    gold_docs: list[dict[str, Any]],
    results: list[dict[str, Any]],
    candidate_sets: dict[str, set[tuple[str,str]]],
    topics: dict[str, str | None]
) -> dict[str, Any]:
    gold = {}
    for row in gold_docs:
        did=row["doc_id"]
        if did in gold:
            raise ValueError("duplicate gold doc " + did)
        pairs=parse_pair_list(row["positive_pairs"])
        if not pairs.issubset(candidate_sets[did]):
            raise ValueError("Gold pair outside candidate universe")
        gold[did] = pairs
    if set(gold) != set(candidate_sets):
        raise ValueError("Gold documents do not exactly cover the candidate manifest")
    arms: dict[str, dict[str, set[tuple[str,str]]]] = defaultdict(dict)
    for res in results:
        did,arm=res["doc_id"],res["arm"]
        if did not in gold:
            raise ValueError("Missing gold doc " + did)
        if did in arms[arm]:
            raise ValueError("Duplicate result arm/document")
        p=parse_pair_list(res["predicted_pairs"])
        if not p.issubset(candidate_sets[did]):
            raise ValueError("Prediction outside candidate universe")
        arms[arm][did]=p
    if not arms:
        raise ValueError("no results")
    for arm, by_doc in arms.items():
        if set(by_doc) != set(gold):
            raise ValueError("Arm %s does not cover all gold documents" % arm)
    summary={}
    per_topic: dict[str, dict[str, tuple[int,int,int]]] = defaultdict(dict)
    for arm,by_doc in sorted(arms.items()):
        rows=[]
        trows: dict[str, list[tuple[int,int,int]]]=defaultdict(list)
        for did,g in gold.items():
            pred=by_doc[did]
            triple=(len(pred & g),len(pred),len(g))
            rows.append(triple)
            trows[str(topics.get(did) or "UNKNOWN")].append(triple)
        summary[arm]=accumulate(rows)
        for topic,xx in trows.items():
            z=accumulate(xx)
            per_topic[topic][arm]=(z["TP"],z["predicted"],z["gold"])
    comparisons={}
    pairs_cmp=[]
    if "A" in arms and "R" in arms: pairs_cmp.append(("A","R"))
    if "A" in arms and "B" in arms: pairs_cmp.append(("A","B"))
    if "P" in arms and "B" in arms: pairs_cmp.append(("P","B"))
    for a,b in pairs_cmp:
        added=set();removed=set(); correct_added=correct_removed=0
        for did in gold:
            plus=arms[a][did]-arms[b][did]
            minus=arms[b][did]-arms[a][did]
            correct_added+=len(plus & gold[did]);correct_removed+=len(minus & gold[did])
            added.update((did,*p) for p in plus)
            removed.update((did,*p) for p in minus)
        base=summary[b]
        delta_tp=correct_added-correct_removed
        delta_pred=len(added)-len(removed)
        utility=2*delta_tp-float(base["F1"])*delta_pred
        comparisons[a+"_vs_"+b]={
            "delta_F1_points":100*(float(summary[a]["F1"])-float(base["F1"])),
            "added_total":len(added),"added_TP":correct_added,
            "added_FP":len(added)-correct_added,
            "added_precision":correct_added/len(added) if added else None,
            "removed_total":len(removed),"removed_TP":correct_removed,
            "removed_FP":len(removed)-correct_removed,
            "delta_TP":delta_tp,"delta_predicted":delta_pred,
            "utility":utility,
            "ci":bootstrap_diff_by_topic(per_topic,a,b),
        }
    return {
        "warning":"Gold is evaluation-only. These are measurements only if inputs are actual model responses.",
        "results_by_arm":summary,
        "comparisons":comparisons,
        "topics":{topic:{arm:measures(*triple) for arm,triple in amap.items()}
                  for topic,amap in per_topic.items()},
    }
