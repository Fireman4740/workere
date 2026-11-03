"""Evaluation is deliberately separate from inference and request construction."""
import json
from pathlib import Path
from .contracts import canonical, digest, require


def freeze_predictions(predictions, provenance, path):
    payload = dict(predictions=predictions, provenance=provenance, gold_opened=False)
    data = canonical(payload)
    # Exclusive creation prevents replacing predictions after looking at labels.
    with Path(path).open("xb") as f:
        f.write(data)
    return digest(payload)


def metrics(predicted, gold, candidates):
    predicted, gold, candidates = set(predicted), set(gold), set(candidates)
    require(predicted <= candidates and gold <= candidates, "outside evaluation candidates")
    tp, fp, fn = len(predicted & gold), len(predicted - gold), len(gold - predicted)
    denominator = 2 * tp + fp + fn
    return dict(TP=tp, FP=fp, FN=fn, micro_F1=2 * tp / denominator if denominator else 0.,
                support=len(gold), predictions=len(predicted), zero_denominator=0.)


def evaluate_frozen(path, expected_hash, gold, candidate_set):
    payload = json.loads(Path(path).read_text())
    require(digest(payload) == expected_hash and payload["gold_opened"] is False, "freeze mismatch")
    predicted = {tuple(p) for p in payload["predictions"]}
    return metrics(predicted, gold, candidate_set)


def paired_group_bootstrap(first, second, gold, candidates, groups, *, seed=20261103, draws=10000, contrasts=1):
    """Fixed configurations only; Bonferroni family interval for registered contrasts."""
    import random
    require(draws >= 100 and contrasts >= 1, "invalid bootstrap budget")
    require(set(groups) == set(candidates), "every candidate needs a group")
    require(set(first) <= set(candidates) and set(second) <= set(candidates) and set(gold) <= set(candidates), "outside bootstrap candidates")
    names = sorted(set(groups.values()))
    require(len(names) >= 2, "group bootstrap requires multiple groups")
    def counts(prediction, group):
        cs = {p for p in candidates if groups[p] == group}
        m = metrics(set(prediction)&cs,set(gold)&cs,cs)
        return [m[k] for k in ("TP","FP","FN")]
    bank = [{g:counts(pred,g) for g in names} for pred in (first,second)]
    def score(count):
        a,b,c = count
        return 2*a/(2*a+b+c) if 2*a+b+c else 0.
    rng = random.Random(seed)
    deltas = []
    for _ in range(draws):
        sample = rng.choices(names,k=len(names))
        values = [score([sum(bs[g][i] for g in sample) for i in range(3)]) for bs in bank]
        deltas.append(values[1]-values[0])
    deltas.sort()
    tail = .025/contrasts
    return dict(delta=metrics(second,gold,candidates)["micro_F1"]-metrics(first,gold,candidates)["micro_F1"],
                interval=[deltas[int(draws*tail)], deltas[min(draws-1,int(draws*(1-tail)))]],
                groups=len(names), draws=draws, seed=seed, family_contrasts=contrasts)
