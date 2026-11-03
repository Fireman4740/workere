"""Exact expected micro-F1 decoding on a fixed batch; no labels accepted."""
import math
from .contracts import require


def decode(hypotheses, weights=None, *, neutral_cardinalities=False):
    hypotheses = [set(h) for h in hypotheses]
    if not hypotheses:
        return dict(predictions=[], expected_utility=1., k=0, union_size=0)
    weights = list(weights) if weights is not None else [1.] * len(hypotheses)
    require(len(weights) == len(hypotheses) and all(math.isfinite(w) and w >= 0 for w in weights)
            and sum(weights) > 0, "invalid reference weights")
    total = sum(weights)
    weights = [w / total for w in weights]
    union = sorted(set().union(*hypotheses))
    sizes = [len(h) for h in hypotheses]
    if neutral_cardinalities:
        sizes = [sum(w * len(h) for w, h in zip(weights, hypotheses))] * len(hypotheses)
    best = dict(predictions=[], expected_utility=sum(w for w, h in zip(weights, hypotheses) if not h), k=0, union_size=len(union))
    for k in range(1, len(union) + 1):
        scores = {i: sum(w * 2 / (k + n) for w, h, n in zip(weights, hypotheses, sizes) if i in h) for i in union}
        selected = sorted(union, key=lambda i: (-scores[i], i))[:k]
        utility = sum(scores[i] for i in selected)
        if utility > best["expected_utility"] + 1e-12:
            best.update(predictions=sorted(selected), expected_utility=utility, k=k)
    return best


def family_weights(families):
    """Each reader family gets equal total mass regardless of sample count."""
    from collections import Counter
    counts = Counter(families)
    return [1 / (len(counts) * counts[f]) for f in families]


def batch_hypotheses(readings, documents):
    """Each calendar reference contains every document exactly once."""
    by = {}
    for r in readings:
        require(r["status"] == "valid", "invalid reference")
        key = (r["family"], r["calendar_id"], r["slot_id"], r["replicate_id"])
        bank = by.setdefault(key, {})
        require(r["doc_id"] not in bank and r["doc_id"] in documents, "duplicate or unknown document")
        bank[r["doc_id"]] = documents[r["doc_id"]].pairs(r["relations"])
    hypotheses, families = [], []
    for key, bank in sorted(by.items()):
        require(set(bank) == set(documents), "incomplete batch reference")
        hypotheses.append({(documents[d].corpus_revision, d, *p) for d, ps in bank.items() for p in ps})
        families.append(key[0])
    return hypotheses, family_weights(families)
