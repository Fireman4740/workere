"""Optional external-model plugin. Importing it never calls an API.

Only the two explicitly supported :free routes are allowed. The execution YAML must enable
network access. A paid provider must use a separately registered may_charge=True plugin.
"""
from benchmark.examples.openrouter_decisions import MODELS, call, decode, request_body
from benchmark.registry import register
from benchmark.schema import Relation, probability


@register("relation_predictor", "openrouter_free", version="1", requires_network=True)
class OpenRouterDecisions:
    def __init__(self, model, expected_version, threshold=.5):
        if model not in MODELS or not isinstance(expected_version, str) or not expected_version:
            raise ValueError("pin one supported free model and its expected served version")
        probability(threshold)
        if threshold is None:
            raise ValueError("threshold must be numeric")
        self.model, self.version, self.threshold = model, expected_version, threshold

    def predict(self, document, entities, candidates, context):
        by_id, relations = {e.id: e for e in entities}, []
        for source, target in candidates:
            def description(mid):
                e = by_id[mid]
                return f"{mid} at tokens {list(e.token_indices)}: " + " ".join(document.tokens[i] for i in e.token_indices)
            response = call(request_body(self.model, document.text, description(source), description(target)))
            usage = response.get("usage", {})
            context.record_usage(cost_usd=usage.get("cost"), model=response.get("model"),
                                 source=source, target=target, response=response)
            if response.get("model") != self.version:
                raise ValueError(f"served model version changed: {response.get('model')}")
            if usage.get("cost", 0) not in (0, None):
                raise ValueError("free model route returned a nonzero cost")
            label = decode(self.model, response, self.threshold)
            answer = response["answers"]
            confidence = (max(answer["relation"]["probabilities"].values()) if self.model == MODELS[0]
                          else max(answer[key]["noul"] for key in ("forward", "reverse")))
            relations.append(Relation(source, target, label, confidence))
        return tuple(relations)
