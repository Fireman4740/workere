"""Configurable pair-level Noul producer registered as an external benchmark component."""
from benchmark.registry import register
from benchmark.schema import Relation
from pipelines.jev.client import DecisionClient, is_free
from pipelines.jev.postprocessor import noul_label
from pipelines.jev.state_builder import NOUL_INSTRUCTIONS, build_state, noul_question


@register("relation_predictor", "jev_decision", version="1", requires_network=True, may_charge=True)
class JevDecisionPredictor:
    @staticmethod
    def access_requirements(params):
        client = params.get("client", {})
        mock = client.get("mode", "mock") == "mock"
        return {"requires_network": not mock, "may_charge": not mock and not is_free(client),
                "mode": "mock" if mock else "inference"}

    def __init__(self, client, threshold=.5, instructions=NOUL_INSTRUCTIONS):
        self.client, self.threshold, self.instructions = DecisionClient(**client), threshold, instructions

    def predict(self, document, entities, candidates, context):
        entities = {e.id: e for e in entities}
        question, out, scores = noul_question(self.instructions), [], []
        for source, target in candidates:
            state = build_state(document, entities[source], entities[target])

            def record(event):
                context.record_usage(cost_usd=event["cost_usd"], source=source, target=target, **{
                    key: value for key, value in event.items() if key != "cost_usd"})

            packet = self.client.decide(state, question, on_attempt=record)
            value = packet["response"]["answers"]["causal"]["noul"]
            label = noul_label(value, self.threshold)
            out.append(Relation(source, target, label, value if label == "causal" else 1 - value))
            scores.append({"source": source, "target": target, "positive_probability": value,
                           "cache_key": packet["cache_key"], "cache_hit": packet["cache_hit"],
                           "served_model": packet["response"]["model"]})
            if len(scores) % 25 == 0 or len(scores) == len(candidates):
                print(f"decisions document={document.id} completed={len(scores)}/{len(candidates)} "
                      f"api_attempts={self.client.calls} cost_usd={context.cost_usd}", flush=True)
        context.prediction_metadata.update(decision_scores=scores, decision_type="noul", threshold=self.threshold,
                                           direction_predicted=False)
        return tuple(out)

    def close(self):
        self.client.close()
