"""Noul is causal existence only: never invent a causal direction."""
from benchmark.labels import BINARY_POS, NOREL
from benchmark.schema import probability


def noul_label(positive_probability, threshold=.5):
    probability(positive_probability)
    probability(threshold)
    if positive_probability is None or threshold is None:
        raise ValueError("probability and threshold must be numeric")
    return BINARY_POS if positive_probability >= threshold else NOREL
