"""D0: paired reader errors and oracle union ceiling, not a deployed selector."""
from .evaluation import metrics
from .contracts import require


def diagnose(systems, gold, candidates):
    require(bool(systems), "no comparable systems")
    candidates, gold = set(candidates), set(gold)
    systems = {k:set(v) for k,v in systems.items()}
    scores = {k:metrics(v, gold, candidates) for k,v in systems.items()}
    union = set().union(*systems.values())
    recall = len(union & gold) / len(gold) if gold else 0.
    return dict(systems=scores, union_recall=recall,
        oracle_union_F1_ceiling=2*recall/(1+recall) if gold else 0.,
        oracle_only=True,
        unique_positives={k:sorted(v - set().union(*(w for n,w in systems.items() if n != k))) for k,v in systems.items()},
        unique_TP={k:sorted((v & gold) - set().union(*(w for n,w in systems.items() if n != k))) for k,v in systems.items()},
        common_FN=sorted(gold-union), common_FP=sorted(set.intersection(*systems.values())-gold),
        link_counts={k:len(v) for k,v in systems.items()})
