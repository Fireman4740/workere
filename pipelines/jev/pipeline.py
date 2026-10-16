"""Configurable pair-level Noul producer registered as an external benchmark component."""
import threading
from concurrent.futures import ThreadPoolExecutor

from benchmark.registry import register
from benchmark.labels import CAUSES, CAUSED_BY, NOREL
from benchmark.schema import Relation
from pipelines.jev.client import DecisionClient, is_free
from pipelines.jev.postprocessor import noul_label
from pipelines.jev.state_builder import build_state, questions_from_params, request_pairs


@register("relation_predictor", "jev_decision", version="1", requires_network=True, may_charge=True)
class JevDecisionPredictor:
    @staticmethod
    def access_requirements(params):
        client = params.get("client", {})
        mock = client.get("mode", "mock") == "mock"
        return {"requires_network": not mock, "may_charge": not mock and not is_free(client),
                "mode": "mock" if mock else "inference"}

    def __init__(self, client, threshold=.5, instructions=None, variant="noul", context_style="document",
                 criteria=None, extra_questions=None, max_workers=1, unordered=False, demonstrations=None, preflight=None):
        # Reject incompatible question types before any model calls.
        if variant == "choice" and client["model"].startswith("respan/span-"):
            raise ValueError("Span supports noul only; choose Mercury for the Choice experiment")
        if type(max_workers) is not int or max_workers < 1:
            raise ValueError("max_workers must be a positive integer")
        self.variant, self.context_style, self.max_workers = variant, context_style, max_workers
        if type(unordered) is not bool or (unordered and variant != "noul"):
            raise ValueError("unordered requests require a boolean flag and Noul")
        self.unordered = unordered
        self.demonstrations = None
        if demonstrations:
            from pipelines.jev.demonstrations import RetrievedDemonstrations
            self.demonstrations = RetrievedDemonstrations(**demonstrations)
        self.question = questions_from_params({"variant": variant, "instructions": instructions,
                                               "criteria": criteria, "extra_questions": extra_questions})
        noul_label(.5, threshold)  # validate before allocation or inference
        self.client, self.threshold, self.instructions = DecisionClient(**client), threshold, instructions
        if preflight:
            from pipelines.jev.preflight import check_multi_questions
            check_multi_questions(self.client, self.question, preflight["path"], self.max_workers)

    def _decide(self, document, entities, source, target, context, lock):
        state = build_state(document, entities[source], entities[target], self.context_style)
        if self.demonstrations:
            state = self.demonstrations.prefix(document, entities[source], entities[target]) + state

        def record(event):
            decision = {}
            answer = ((event.get("response") or {}).get("answers") or {}).get("causal", {})
            if self.variant == "noul" and not event.get("error") and answer.get("noul") is not None:
                p = answer["noul"]
                label = noul_label(p, self.threshold)
                decision = {"positive_probability": p, "predicted_label": label, "threshold": self.threshold,
                            "class_confidence": p if label == "causal" else 1 - p}
            with lock:  # Context and the decision journal are shared by all worker threads.
                context.record_usage(cost_usd=event["cost_usd"], source=source, target=target, **decision, **{
                    key: value for key, value in event.items() if key != "cost_usd"})

        packet = self.client.decide(state, self.question, on_attempt=record)
        answers = packet["response"]["answers"]
        answer = answers["causal" if self.variant == "noul" else "relation"]
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
        score = {"source": source, "target": target, "positive_probability": value, **extra,
                 "answers": {key: a.get("noul", a.get("choice")) for key, a in answers.items()
                             if key in self.question},
                 "cache_key": packet["cache_key"], "cache_hit": packet["cache_hit"],
                 "served_model": packet["response"]["model"]}
        return Relation(source, target, label, confidence), score

    def predict(self, document, entities, candidates, context):
        queries = request_pairs(document, entities, candidates, self.unordered)
        entities = {e.id: e for e in entities}
        lock, done = threading.Lock(), [0]

        def one(pair):
            result = self._decide(document, entities, pair[0], pair[1], context, lock)
            with lock:
                done[0] += 1
                if done[0] % 25 == 0 or done[0] == len(queries):
                    print(f"decisions document={document.id} completed={done[0]}/{len(queries)} "
                          f"api_attempts={self.client.calls} cost_usd={context.cost_usd}", flush=True)
            return result

        if self.max_workers == 1:
            results = [one(pair) for pair in queries]
        else:
            # map() keeps candidate order; the first worker exception propagates unchanged.
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                results = list(pool.map(one, queries))
        if self.unordered:
            lookup = {frozenset((r.source, r.target)): (r, score) for r, score in results}
            expanded = []
            for a, b in candidates:
                relation, score = lookup[frozenset((a, b))]
                expanded.append((Relation(a, b, relation.label, relation.confidence),
                    {**score, "query_source": relation.source, "query_target": relation.target,
                     "source": a, "target": b}))
            results = expanded
        out = tuple(relation for relation, _ in results)
        context.prediction_metadata.update(decision_scores=[score for _, score in results], decision_type=self.variant,
                                           threshold=self.threshold if self.variant == "noul" else None,
                                           direction_predicted=self.variant == "choice", context_style=self.context_style,
                                           questions=sorted(self.question), unordered_requests=self.unordered,
                                           query_count=len(queries))
        return out

    def close(self):
        self.client.close()
