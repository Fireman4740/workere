"""Relation predictors driven by a Harness² component tree; the frozen R3 systems otherwise.

`luna_harness` and `decision_harness` behave byte-identically to `luna_document` and
`jev_decision` when given the base tree. Their per-document trace (`h2_trace`) is what the
harness editor reads as the rollout's trajectory; it never contains annotations.
"""
import json
import random
import threading

from benchmark.data import parse_annotations
from benchmark.labels import BINARY_POS, NOREL, is_norel
from benchmark.registry import register
from benchmark.schema import Relation
from benchmark.voting import aggregate
from eci.parse import parse_triples
from pipelines.harness2_eci import postprocess
from pipelines.harness2_eci.surface import decision_question, luna_request_parts
from pipelines.jev.llm import LunaDocumentPredictor, marked_text
from pipelines.jev.pipeline import JevDecisionPredictor
from pipelines.jev.postprocessor import noul_label

TRACE_RESPONSE_CHARS = 12_000


def read_harness(harness_dir):
    from harness2.core import tree as harness_tree
    from harness2.substrate.opencode import OPENCODE_SURFACE
    return harness_tree.read_tree(OPENCODE_SURFACE, harness_dir)


def _mentions(document, entities):
    return [{"id": e.id, "surface": " ".join(document.tokens[i] for i in e.token_indices),
             "sentence": document.sentence_of(e.token_indices)} for e in entities]


@register("relation_predictor", "luna_harness", version="1", requires_network=True, may_charge=True)
class LunaHarnessPredictor(LunaDocumentPredictor):
    def __init__(self, client, path, selection_path, repo_id, revision, harness_dir, passes=7):
        super().__init__(client, path, selection_path, repo_id, revision, passes)
        self.tree = read_harness(harness_dir)
        self.system, self.prefix, self.suffix = luna_request_parts(self.tree, self.prompt)

    def _user(self, text):
        return self.prefix + "Text:\n" + text + self.suffix

    def predict(self, document, entities, candidates, context):
        eligible = [d for d in self.bank if d["id"].split("_")[0] != document.id.split("_")[0]]
        if self.client.mode != "mock" and len(eligible) < self.passes:
            raise ValueError("insufficient demonstrations outside evaluation topic")
        selected = (random.Random(f"{context.seed}::{document.id}").sample(eligible, self.passes)
                    if eligible else [None] * self.passes)
        lock, prepared = threading.Lock(), []
        for index, demo in enumerate(selected):
            messages = [{"role": "system", "content": self.system}]
            if demo:
                answer = [{"pair": f"{a},{b}", "label": lab} for a, lab, b in parse_annotations(demo["annots"])
                          if not is_norel(lab)]
                messages += [{"role": "user", "content": self._user(demo["text"])},
                             {"role": "assistant", "content": json.dumps(answer)}]
            messages.append({"role": "user", "content": self._user(marked_text(document, entities))})

            def record(event, index=index, demo=demo):
                with lock:
                    context.record_usage(cost_usd=event["cost_usd"], pass_index=index,
                                         demonstration_id=demo["id"] if demo else None,
                                         **{k: v for k, v in event.items() if k != "cost_usd"})
            prepared.append((messages, record))
        packets = self.client.decide_many([m for m, _ in prepared], [r for _, r in prepared])
        allowed, passes, trace = set(candidates), [], []
        for index, packet in enumerate(packets):
            content = packet["response"]["choices"][0]["message"]["content"]
            triples = [(a, lab, b) for a, lab, b in parse_triples(content)
                       if (a, b) in allowed and lab in {"causes", "causedby", "norel"}]
            passes.append(triples)
            trace.append(f"[ACTION] pass {index + 1}/{self.passes}: one request (system + one worked "
                         f"example from another topic + this article)\n[OBSERVATION] response "
                         f"({len(content)} chars, {sum(not is_norel(t[1]) for t in triples)} causal pairs "
                         f"kept):\n{content[:TRACE_RESPONSE_CHARS]}")
        final, stats = aggregate(passes, NOREL, context.rng)
        default = {(a, b): lab for a, lab, b in final}
        pairs = []
        for a, b in candidates:
            votes = stats.get(f"{a},{b}", {}).get("votes", [NOREL] * self.passes)
            pairs.append({"source": a, "target": b, "default_label": default.get((a, b), NOREL),
                          "probability": sum(not is_norel(v) for v in votes) / self.passes, "votes": votes})
        labels, log = postprocess.apply(self.tree, "luna", {
            "document_id": document.id, "executor": "luna", "mentions": _mentions(document, entities),
            "pairs": pairs})
        if log:
            trace.append(log)
        context.prediction_metadata.update(
            decision_scores=[{"source": p["source"], "target": p["target"], "positive_probability": p["probability"]}
                             for p in pairs],
            passes=self.passes, demonstration_ids=[d["id"] for d in selected if d],
            model=self.client.identity["model"], h2_trace="\n\n".join(trace))
        return tuple(Relation(a, b, labels[f"{a},{b}"]) for a, b in candidates)


@register("relation_predictor", "decision_harness", version="1", requires_network=True, may_charge=True)
class DecisionHarnessPredictor(JevDecisionPredictor):
    def __init__(self, client, harness_dir, threshold=.5, context_style="document", max_workers=1):
        super().__init__(client, threshold=threshold, context_style=context_style, max_workers=max_workers)
        self.tree = read_harness(harness_dir)
        self.question = decision_question(self.tree)

    def predict(self, document, entities, candidates, context):
        super().predict(document, entities, candidates, context)
        scores = context.prediction_metadata["decision_scores"]
        pairs = [{"source": s["source"], "target": s["target"], "probability": s["positive_probability"],
                  "default_label": noul_label(s["positive_probability"], self.threshold)} for s in scores]
        labels, log = postprocess.apply(self.tree, "decision", {
            "document_id": document.id, "executor": "decision", "mentions": _mentions(document, entities),
            "pairs": pairs})
        ranked = sorted(pairs, key=lambda p: -p["probability"])
        lines = [f"{p['source']}->{p['target']} p={p['probability']:.3f}" for p in ranked[:150]]
        trace = [f"[ACTION] {len(pairs)} independent decision requests, one per ordered mention pair, each "
                 f"with the full article, the two mentions and the question:\n"
                 f"{json.dumps(self.question, ensure_ascii=False)}",
                 f"[OBSERVATION] positive probabilities, highest first (threshold {self.threshold}; "
                 f"{sum(p['default_label'] == BINARY_POS for p in pairs)} pairs at or above it):\n"
                 + "\n".join(lines)]
        if log:
            trace.append(log)
        context.prediction_metadata["h2_trace"] = "\n\n".join(trace)
        return tuple(Relation(p["source"], p["target"], labels[f"{p['source']},{p['target']}"])
                     for p in pairs)
