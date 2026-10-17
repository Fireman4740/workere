"""Document-level Luna reference using the historical prompt, parser and voting rule.

One gold-answer demonstration per pass; no synthetic CoT demonstration or historical
DeepSeek parity claim. Evaluation annotations never enter the model-facing interface.
"""
import json
import hashlib
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from benchmark.data import parse_annotations
from benchmark.labels import NOREL, is_norel
from benchmark.registry import register
from benchmark.schema import Relation
from benchmark.voting import aggregate
from eci.parse import parse_triples
from pipelines.jev.client import DecisionClient, DecisionError, BudgetExceeded

_IN_FLIGHT = threading.BoundedSemaphore(128)
_BANK_LOCK = threading.Lock()
_BANKS = {}


def frozen_demonstration_bank(path, repo_id, revision):
    data = json.loads(Path(path).read_text())
    if data.get("schema") != "workere-frozen-demo-bank/v1" or data.get("repo_id") != repo_id or data.get("revision") != revision:
        raise ValueError("frozen demonstration bank identity mismatch")
    rows = data["documents"]
    if len(rows) != 20 or len({d["id"] for d in rows}) != 20:
        raise ValueError("frozen demonstration bank must contain 20 distinct documents")
    return rows


def marked_text(document, entities):
    starts, ends = {}, {}
    for e in entities:
        # Keep every event's exact token indices, including discontinuous spans.
        for i in e.token_indices:
            if i - 1 not in e.token_indices:
                starts.setdefault(i, []).append(e.id)
            if i + 1 not in e.token_indices:
                ends.setdefault(i, []).append(e.id)
    return " ".join("".join(f"<{mid} " for mid in starts.get(i, [])) + token
                    + ">" * len(ends.get(i, [])) for i, token in enumerate(document.tokens))


def demonstration_bank(repo_id, revision, selection_path):
    key = (repo_id, revision, selection_path)
    with _BANK_LOCK:
        if key not in _BANKS:
            from datasets import load_dataset
            selection = json.loads(Path(selection_path).read_text())
            excluded = set(selection["diagnostic_d15"]) | {d["id"] for d in selection["confirmation"]["docs"]}
            rows = load_dataset(repo_id, split="train", revision=revision)
            # Exclusion happens before accessing any relation annotations.
            eligible = sorted((r for r in rows if r["id"] not in excluded), key=lambda r: r["id"])
            random.Random(42).shuffle(eligible)
            _BANKS[key] = eligible[:20]
        return _BANKS[key]


class ChatClient(DecisionClient):
    def __init__(self, *, reasoning_effort="high", max_output_tokens=16384, **kwargs):
        self.reasoning_effort, self.max_output = reasoning_effort, max_output_tokens
        super().__init__(**kwargs)

    def validate_request(self, messages, questions):
        body = {"model": self.identity["model"], "messages": messages,
                "reasoning": {"effort": self.reasoning_effort}, "max_tokens": self.max_output}
        bound = len(json.dumps(body, ensure_ascii=False).encode()) + 1024
        if bound + self.max_output > self.max_context:
            raise DecisionError("chat context overflow; no truncation applied")
        if self.mode != "mock" and bound * self.input_price / 1e6 + self.max_output * .25 / 1e6 > self.reservation:
            raise BudgetExceeded("chat per-call reservation below conservative token cost")
        return body, bound

    def _send(self, body):
        if self.mode == "mock":
            return {"model": "mock/" + self.identity["model_version"], "provider": "mock",
                    "choices": [{"finish_reason": "stop", "message": {"content": "[]"}}], "usage": {"cost": 0}}
        if self.identity["model"].endswith(":batch"):
            raise DecisionError("a :batch model must use decide_many through /api/v1/batches")
        with _IN_FLIGHT:
            return super()._send(body)

    def _validate_response(self, body, response):
        if not isinstance(response, dict) or not isinstance(response.get("model"), str):
            raise DecisionError("chat response lacks served model")
        if self.mode != "mock" and response["model"] not in {
                "openai/gpt-6-luna", "openai/gpt-6-luna:batch", "openai/gpt-6-luna-20260922"}:
            raise DecisionError("unexpected chat model: " + response["model"])
        choice = response.get("choices", [{}])[0]
        if choice.get("finish_reason") != "stop":
            raise DecisionError("incomplete chat generation: " + str(choice.get("finish_reason")))
        content = choice.get("message", {}).get("content")
        if not isinstance(content, str):
            raise DecisionError("chat response lacks text content")
        parse_triples(content)  # malformed answers are retried and never scored as negatives


@register("relation_predictor", "luna_document", version="1", requires_network=True, may_charge=True)
class LunaDocumentPredictor:
    @staticmethod
    def execution_policy(params):
        mock = params.get("client", {}).get("mode") == "mock"
        return {"requires_network": not mock, "may_charge": not mock, "mode": "mock" if mock else "inference"}

    def __init__(self, client, path, selection_path, repo_id, revision, passes=7, demo_bank_path=None, demo_selection="random"):
        import yaml
        from pipelines.jev.batch import BatchChatClient
        self.client = BatchChatClient(**client)
        self.prompt = yaml.safe_load(Path(path).read_text())
        if demo_selection not in {"random", "contrastive_structural"}:
            raise ValueError("unknown demonstration selection")
        self.passes, self.bank, self.demo_selection = passes, [], demo_selection
        if self.client.mode != "mock":
            self.bank = (frozen_demonstration_bank(demo_bank_path, repo_id, revision) if demo_bank_path
                         else demonstration_bank(repo_id, revision, selection_path))
        print("LUNA_GUIDANCE_CONFIG " + json.dumps({"passes": passes, "prompt_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "demo_bank_sha256": hashlib.sha256(Path(demo_bank_path).read_bytes()).hexdigest() if demo_bank_path else None,
            "demonstration_selection": demo_selection + "_outside_topic", "demonstration_count": len(self.bank)}), flush=True)

    def predict(self, document, entities, candidates, context):
        user = self.prompt["user_template"].format(doc_text=marked_text(document, entities), pair_lines="", doc_id=document.id)
        eligible = [d for d in self.bank if d["id"].split("_")[0] != document.id.split("_")[0]]
        if self.client.mode != "mock" and len(eligible) < self.passes:
            raise ValueError("insufficient demonstrations outside evaluation topic")
        if eligible and self.demo_selection == "contrastive_structural":
            from pipelines.jev.contrastive import select_demonstrations
            selected = select_demonstrations(document, entities, self.bank, self.passes)
        else:
            selected = random.Random(f"{context.seed}::{document.id}").sample(eligible, self.passes) if eligible else [None] * self.passes
        lock = threading.Lock()

        def prepare(item):
            index, demo = item
            messages = [{"role": "system", "content": self.prompt["system"]}]
            if demo:
                answer = [{"pair": f"{a},{b}", "label": lab} for a, lab, b in parse_annotations(demo["annots"]) if not is_norel(lab)]
                if self.demo_selection == "contrastive_structural":
                    from pipelines.jev.contrastive import negative_pairs
                    answer += [{"pair": f"{x},{y}", "label": "norel"} for a, b in negative_pairs(demo) for x, y in [(a, b), (b, a)]]
                messages += [{"role": "user", "content": self.prompt["user_template"].format(doc_text=demo["text"], pair_lines="", doc_id=demo["id"])},
                             {"role": "assistant", "content": json.dumps(answer)}]
            messages.append({"role": "user", "content": user})

            def record(event):
                with lock:
                    context.record_usage(cost_usd=event["cost_usd"], pass_index=index,
                        demonstration_id=demo["id"] if demo else None,
                        **{k: v for k, v in event.items() if k != "cost_usd"})
            return messages, record

        prepared = [prepare(item) for item in enumerate(selected)]
        packets = self.client.decide_many([m for m, _ in prepared], [callback for _, callback in prepared])
        passes = []
        for index, packet in enumerate(packets):
            triples = parse_triples(packet["response"]["choices"][0]["message"]["content"])
            allowed = set(candidates)
            triples = [(a, lab, b) for a, lab, b in triples if (a, b) in allowed and lab in {"causes", "causedby", "norel"}]
            print(f"llm document={document.id} pass={index + 1}/{self.passes} model={packet['response']['model']} "
                  f"cost_usd={context.cost_usd}", flush=True)
            passes.append(triples)
        final, stats = aggregate(passes, NOREL, context.rng)
        labels = {(a, b): lab for a, lab, b in final}
        scores = []
        for a, b in candidates:
            counts = stats.get(f"{a},{b}", {}).get("vote_counts", {})
            p = sum(v for lab, v in counts.items() if not is_norel(lab)) / self.passes
            scores.append({"source": a, "target": b, "positive_probability": p})
        context.prediction_metadata.update(decision_scores=scores, direction_predicted=True,
            pass_triples=passes,
            demonstration_ids=[d["id"] for d in selected if d], passes=self.passes,
            demonstration_selection=self.demo_selection,
            model=self.client.identity["model"], reasoning_effort="high",
            demonstration_style=("gold_positive_and_explicit_negative_answers" if self.demo_selection == "contrastive_structural"
                                 else "gold_answer_only_no_synthesized_rationale"), historical_parity="not_claimed")
        return tuple(Relation(a, b, labels.get((a, b), NOREL)) for a, b in candidates)

    def close(self):
        self.client.close()
