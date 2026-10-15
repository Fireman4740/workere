"""Configurable pair-level Noul producer registered as an external benchmark component."""
from benchmark.registry import register
from benchmark.labels import CAUSES, CAUSED_BY, NOREL
from benchmark.schema import Relation
from pipelines.jev.client import DecisionClient, is_free
from pipelines.jev.postprocessor import noul_label
from pipelines.jev.state_builder import NOUL_INSTRUCTIONS, build_state, questions_for


@register("relation_predictor", "jev_decision", version="1", requires_network=True, may_charge=True)
class JevDecisionPredictor:
    @staticmethod
    def access_requirements(params):
        client = params.get("client", {})
        mock = client.get("mode", "mock") == "mock"
        return {"requires_network": not mock, "may_charge": not mock and not is_free(client),
                "mode": "mock" if mock else "inference"}

    def __init__(self, client, threshold=.5, instructions=NOUL_INSTRUCTIONS, variant="noul", context_style="document"):
        # Reject incompatible question types before any model calls.
        if variant == "choice" and client["model"].startswith("respan/span-"):
            raise ValueError("Span supports noul only; choose Mercury for the Choice experiment")
        self.variant, self.context_style = variant, context_style
        self.question = questions_for(variant, instructions)
        noul_label(.5, threshold)  # validate before allocation or inference
        self.client, self.threshold, self.instructions = DecisionClient(**client), threshold, instructions

    def predict(self, document, entities, candidates, context):
        entities = {e.id: e for e in entities}
        question, out, scores = self.question, [], []
        for source, target in candidates:
            state = build_state(document, entities[source], entities[target], self.context_style)

            def record(event):
                decision = {}
                answer = ((event.get("response") or {}).get("answers") or {}).get("causal", {})
                if self.variant == "noul" and not event.get("error") and answer.get("noul") is not None:
                    p = answer["noul"]
                    label = noul_label(p, self.threshold)
                    decision = {"positive_probability": p, "predicted_label": label, "threshold": self.threshold,
                                "class_confidence": p if label == "causal" else 1 - p}
                context.record_usage(cost_usd=event["cost_usd"], source=source, target=target, **decision, **{
                    key: value for key, value in event.items() if key != "cost_usd"})

            packet = self.client.decide(state, question, on_attempt=record)
            answer = packet["response"]["answers"]["causal" if self.variant == "noul" else "relation"]
            extra = {}
            if self.variant == "noul":
                value = answer["noul"]
                label = noul_label(value, self.threshold)
                confidence = value if label == "causal" else 1 - value
            else:
                mapping = {"CAUSES": CAUSES, "CAUSED_BY": CAUSED_BY, "NO_RELATION": NOREL}
                value = 1 - answer["probabilities"]["NO_RELATION"]
                label = mapping[answer["choice"]]
                confidence = answer["probabilities"][answer["choice"]]
                extra = {"class_probabilities": {mapping[k]: p for k, p in answer["probabilities"].items()},
                         "provider_confidence": answer.get("confidence")}
            out.append(Relation(source, target, label, confidence))
            scores.append({"source": source, "target": target, "positive_probability": value, **extra,
                           "cache_key": packet["cache_key"], "cache_hit": packet["cache_hit"],
                           "served_model": packet["response"]["model"]})
            if len(scores) % 25 == 0 or len(scores) == len(candidates):
                print(f"decisions document={document.id} completed={len(scores)}/{len(candidates)} "
                      f"api_attempts={self.client.calls} cost_usd={context.cost_usd}", flush=True)
        context.prediction_metadata.update(decision_scores=scores, decision_type=self.variant,
                                           threshold=self.threshold if self.variant == "noul" else None,
                                           direction_predicted=self.variant == "choice", context_style=self.context_style)
        return tuple(out)

    def close(self):
        self.client.close()
