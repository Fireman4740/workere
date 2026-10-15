"""Noul is causal existence only: never invent a causal direction."""
from benchmark.labels import BINARY_POS, NOREL
from benchmark.schema import probability


def noul_label(positive_probability, threshold=.5):
    probability(positive_probability)
    probability(threshold)
    if positive_probability is None or threshold is None:
        raise ValueError("probability and threshold must be numeric")
    return BINARY_POS if positive_probability >= threshold else NOREL


# Optional component for recorded predictions: unlike a confidence filter, it can
# promote a previously negative decision when its positive-probability threshold falls.
from benchmark.registry import register
from benchmark.schema import Prediction, Relation


@register("postprocessor", "jev_noul_threshold", version="1", mode="replay")
class NoulPostProcessor:
    def __init__(self, threshold=.5):
        noul_label(.5, threshold)
        self.threshold = threshold

    def process(self, document, prediction, context):
        if prediction.metadata.get("decision_type") != "noul":
            raise ValueError("jev_noul_threshold requires recorded Noul probabilities")
        scores = prediction.metadata.get("decision_scores", [])
        expected = {(r.source, r.target) for r in prediction.relations}
        pairs = [(s["source"], s["target"]) for s in scores]
        if len(pairs) != len(set(pairs)) or set(pairs) != expected:
            raise ValueError("probabilities must cover every recorded decision exactly once")
        relations = []
        for score in scores:
            p = score["positive_probability"]
            label = noul_label(p, self.threshold)
            relations.append(Relation(score["source"], score["target"], label,
                                      p if label == BINARY_POS else 1 - p))
        return Prediction(prediction.id, prediction.entities, tuple(relations),
                          {**prediction.metadata, "threshold": self.threshold, "direction_predicted": False})
