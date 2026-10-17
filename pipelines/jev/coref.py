"""Luna explicit event-coreference chains: one batch call per document, no causal question.

The predictor emits `norel` for every candidate (it is not a causal system) and records the
model's mention groups in prediction metadata for offline intrinsic evaluation and for causal
score propagation. Only document text and reference mention ids enter the request.
"""
import json
import re
import threading

from benchmark.labels import NOREL
from benchmark.registry import register
from benchmark.schema import Relation
from pipelines.jev.client import DecisionError
from pipelines.jev.llm import marked_text

SYSTEM = (
    "You are an expert annotator of event coreference in English news (ECB+ guidelines).\n"
    "Event mentions are marked inline as <ID text>. Two mentions corefer only when they denote the "
    "SAME real-world event instance: same action, same time, same place and same participants.\n"
    "Do NOT group: different instances of the same event type (two separate earthquakes), a sub-event "
    "and its larger event, a cause and its effect, a report about an event and the event itself, or "
    "mentions that are merely related. When unsure, keep mentions apart: precision matters more than recall.\n"
    "Reason briefly, then output the final answer as a JSON array of groups, each group a JSON array of "
    "mention ID strings with at least two members. Mentions not listed are singletons. Output [] if no "
    "two mentions corefer."
)
USER = "Document {doc_id}:\n{doc_text}\n\nGroup the coreferent event mentions."


def _last_array(text):
    """The final answer: among balanced arrays of arrays, the one ending last (outermost on ties)."""
    found = []
    for start in (i for i, ch in enumerate(text) if ch == "["):
        depth = 0
        for end in range(start, len(text)):
            depth += text[end] == "["
            depth -= text[end] == "]"
            if depth == 0:
                try:
                    value = json.loads(text[start:end + 1])
                except ValueError:
                    value = None
                if isinstance(value, list) and all(isinstance(g, list) for g in value):
                    found.append((end, -start, value))
                break
    if not found:
        raise ValueError("no JSON array of groups found")
    return max(found, key=lambda f: (f[0], f[1]))[2]


def parse_groups(text, mention_ids):
    groups = _last_array(text)
    seen, out = set(), []
    for group in groups:
        if not all(isinstance(m, (str, int)) for m in group):
            raise ValueError("group members must be mention ids")
        members = [str(m).strip() for m in group]
        unknown = set(members) - set(mention_ids)
        if unknown:
            raise ValueError(f"unknown mention ids: {sorted(unknown)}")
        if seen & set(members) or len(set(members)) != len(members):
            raise ValueError("a mention appears in more than one group")
        seen |= set(members)
        if len(members) >= 2:
            out.append(sorted(members, key=lambda m: (len(m), m)))
    return out


def _client_class():
    from pipelines.jev.batch import BatchChatClient

    class CorefBatchClient(BatchChatClient):
        """Same transport and budget; validates group answers instead of causal triples."""

        def _validate_response(self, body, response):
            try:
                super()._validate_response(body, {**response, "choices": [
                    {**response.get("choices", [{}])[0], "message": {"content": "[]"}}]})
            except (IndexError, AttributeError) as exc:
                raise DecisionError("malformed chat response") from exc
            content = response["choices"][0]["message"].get("content")
            if not isinstance(content, str):
                raise DecisionError("coreference response lacks text content")
            try:
                parse_groups(content, _ids_in(body))
            except ValueError as exc:
                raise DecisionError("unparseable coreference groups: " + str(exc)) from exc

    return CorefBatchClient


def _ids_in(body):
    return set(re.findall(r"<(\d+) ", body["messages"][-1]["content"]))


@register("relation_predictor", "luna_coref", version="1", requires_network=True, may_charge=True)
class LunaCorefPredictor:
    @staticmethod
    def execution_policy(params):
        mock = params.get("client", {}).get("mode") == "mock"
        return {"requires_network": not mock, "may_charge": not mock, "mode": "mock" if mock else "inference"}

    def __init__(self, client):
        self.client = _client_class()(**client)
        self.lock = threading.Lock()

    def predict(self, document, entities, candidates, context):
        ids = {e.id for e in entities}
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": USER.format(doc_id=document.id, doc_text=marked_text(document, entities))}]

        def record(event):
            with self.lock:
                context.record_usage(cost_usd=event["cost_usd"], **{k: v for k, v in event.items() if k != "cost_usd"})

        packet = self.client.decide_many([messages], [record])[0]
        content = packet["response"]["choices"][0]["message"]["content"]
        groups = parse_groups(content, ids) if self.client.mode != "mock" or content.strip() != "[]" else []
        print(f"coref document={document.id} groups={len(groups)} grouped_mentions={sum(map(len, groups))} "
              f"cost_usd={context.cost_usd}", flush=True)
        context.prediction_metadata.update(coref_groups=groups, model=self.client.identity["model"],
                                           reasoning_effort=self.client.reasoning_effort, causal_prediction=False)
        return tuple(Relation(a, b, NOREL) for a, b in candidates)

    def close(self):
        self.client.close()
