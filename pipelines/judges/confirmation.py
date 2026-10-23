"""RES30 confirmation of two DEV44-frozen rules, sharing one inference producer."""
import fcntl
import json
import os
from pathlib import Path
import random
import threading
import time
import uuid

import numpy as np

from benchmark.configuration import read_yaml
from benchmark.io import fingerprint, write_json
from benchmark.schema import Document, Entity
from pipelines.jev.batch import BatchChatClient
from pipelines.jev.llm import LunaDocumentPredictor
from pipelines.jev.pipeline import JevDecisionPredictor
from pipelines.jev.spend import SharedBudget
from pipelines.judges import band_verifier
from pipelines.judges.calibration import load_freeze, rule_scores
from pipelines.judges.common import Dev44, Scorer, Spend, chunks, client_from, f1_arr, run_parallel
from pipelines.judges.driver import preflight

SCHEMA = "workere-confirmation/v1"
_PRINT = threading.Lock()


def emit(tag, value):
    with _PRINT:
        print(tag + " " + json.dumps(value, ensure_ascii=False), flush=True)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False))
    temp.replace(path)


class Cohort(Dev44):
    """Model-facing population: annotations are absent until attach_gold is called."""
    def __init__(self, path):
        public = json.loads(Path(path).read_text())
        if public["fingerprint"] != fingerprint({k: v for k, v in public.items() if k != "fingerprint"}):
            raise ValueError("public cohort fingerprint differs")
        self.public_fingerprint = public["fingerprint"]
        self.suite = {d["id"]: d for d in public["documents"]}
        if len(self.suite) != len(public["documents"]):
            raise ValueError("duplicate cohort documents")
        self.docs = sorted(self.suite)
        self.topic = {d: d.split("_")[0] for d in self.docs}
        self.keys = {d: [tuple(k) for k in self.suite[d]["pair_list"]] for d in self.docs}
        self.by_doc = {d: {k: {} for k in self.keys[d]} for d in self.docs}
        self.fused = {}
        for row in self.suite.values():
            if set(row) & {"gold", "annots", "causal_origin", "coref_clusters"}:
                raise ValueError("annotation field in public cohort")

    def set_baseline(self, doc, rows):
        scores = {(r["a"], r["b"]): {"luna": r["luna"], "d1": r["d1"]} for r in rows}
        if set(scores) != set(self.keys[doc]) or len(scores) != len(rows):
            raise ValueError("baseline scores do not cover candidates exactly")
        self.by_doc[doc] = scores
        self.fused[doc] = self._fused(doc)

    def attach_gold(self, path):
        gold = json.loads(Path(path).read_text())
        if gold["public_fingerprint"] != self.public_fingerprint or set(gold["positive_pairs"]) != set(self.docs):
            raise ValueError("evaluator population differs")
        for d in self.docs:
            positives = {tuple(k) for k in gold["positive_pairs"][d]}
            if not positives <= set(self.keys[d]):
                raise ValueError("gold includes a pair outside the frozen candidates")
            for k in self.keys[d]:
                self.by_doc[d][k]["gold"] = k in positives


class FrozenLuna(LunaDocumentPredictor):
    """Use the unchanged reference predict method with a committed, disjoint demo bank."""
    def __init__(self, client, prompt, bank, passes=7):
        self.client = BatchChatClient(**client)
        self.prompt, self.bank, self.passes = prompt, bank, passes


class Journal:
    def __init__(self, path, doc, kind):
        self.path, self.doc, self.kind = Path(path), doc, kind
        self.seed, self.rng = 42, random.Random(42)
        self.prediction_metadata = {}
        self.cost_usd, self.api_attempts, self.cache_hits = 0., 0, 0
        self.lock = threading.Lock()

    def record_usage(self, cost_usd=None, **event):
        with self.lock:
            self.cost_usd += cost_usd or 0
            self.api_attempts += int(event.get("api_call", False))
            self.cache_hits += int(event.get("cache_hit", False))
            response = event.get("response") or {}
            row = {"doc": self.doc, "kind": self.kind, "cost_usd": cost_usd,
                   "cache_key": event.get("cache_key"), "cache_hit": event.get("cache_hit"),
                   "api_call": event.get("api_call"), "served_model": response.get("model"),
                   "usage": response.get("usage"), "error": event.get("error")}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            if row["error"]:
                emit("CONFIRM_API_ERROR", row)

    def resources(self):
        return {"reported_cost_usd": self.cost_usd, "api_attempts": self.api_attempts, "cache_hits": self.cache_hits}


def load(path):
    cfg = read_yaml(path)
    if cfg.get("schema") != SCHEMA:
        raise ValueError("unsupported confirmation schema")
    base = Path(path).resolve().parent
    for key in ("public_inputs", "evaluation_only", "selection", "frozen_rules", "demo_bank", "luna_prompt"):
        cfg[key] = str((base / cfg[key]).resolve())
    if cfg["variant"] not in {"mean", "logistic"}:
        raise ValueError("unknown frozen combination")
    if cfg["client"]["reasoning_effort"] != "max":
        raise ValueError("DeepSeek max must be requested explicitly")
    if cfg["samples"] != 2 or cfg["band"] != [.25, .55]:
        raise ValueError("confirmation must retain the frozen J2 judge protocol")
    return cfg


def checked_inputs(cfg):
    freeze = load_freeze(cfg["frozen_rules"])
    selection = json.loads(Path(cfg["selection"]).read_text())
    data = Cohort(cfg["public_inputs"])
    if freeze["fingerprint"] != selection["frozen_rules_fingerprint"]:
        raise ValueError("rule selection identity differs")
    if set(data.docs) != {d["id"] for d in selection["documents"]}:
        raise ValueError("target selection differs")
    if set(data.docs) & set(freeze["training_documents"]):
        raise ValueError("target overlaps calibration training")
    bank = json.loads(Path(cfg["demo_bank"]).read_text())["documents"]
    if set(data.docs) & {d["id"] for d in bank}:
        raise ValueError("target document was a Luna demonstration")
    return data, freeze, bank


def plan(cfg):
    data, freeze, _ = checked_inputs(cfg)
    return {"schema": "workere-confirmation-plan/v1", "variant": cfg["variant"],
            "documents": len(data.docs), "topics": len(set(data.topic.values())),
            "ordered_pairs": sum(map(len, data.keys.values())), "public_fingerprint": data.public_fingerprint,
            "frozen_rule": freeze["rules"][cfg["variant"]], "rules_fingerprint": freeze["fingerprint"],
            "budget": cfg["budget"], "shared_inference": True, "reasoning_effort": "max",
            "annotations_read": False}


def shared_signature(cfg, data):
    return fingerprint({"inference_version": 1, "public": data.public_fingerprint,
                        "luna": cfg["luna_client"], "decision": cfg["decision_client"],
                        "judge": cfg["client"], "band": cfg["band"], "samples": cfg["samples"],
                        "pairs_per_call": cfg["pairs_per_call"],
                        "demo_bank": fingerprint(json.loads(Path(cfg["demo_bank"]).read_text())),
                        "luna_prompt": fingerprint(read_yaml(cfg["luna_prompt"])),
                        "judge_system": band_verifier.SYSTEM})


def budget_status(cfg):
    budget = SharedBudget(**cfg["budget"])
    try:
        return budget.status()
    finally:
        budget.close()


def baseline_one(cfg, data, bank, root, signature, doc):
    path = root / "baseline" / (doc + ".json")
    if path.exists():
        result = json.loads(path.read_text())
        if result["signature"] != signature:
            raise ValueError("incompatible baseline cache")
        emit("CONFIRM_BASELINE_REUSED", {"doc": doc})
        return doc, result["scores"]
    source = data.suite[doc]
    document = Document(doc, source["tokens"], source["sentences"], source["lang"])
    entities = tuple(Entity(mid, tuple(span)) for mid, span in zip(source["mentions"], source["spans"]))
    candidates = tuple(data.keys[doc])
    lc = {**cfg["luna_client"], "cache_path": str(Path(cfg["luna_client"]["cache_path"]).expanduser()),
          "shared_budget": cfg["budget"]}
    dc = {**cfg["decision_client"], "cache_path": str(Path(cfg["decision_client"]["cache_path"]).expanduser()),
          "shared_budget": cfg["budget"]}
    luna_context = Journal(root / "receipts" / (doc + "-luna.jsonl"), doc, "luna")
    luna = FrozenLuna(lc, read_yaml(cfg["luna_prompt"]), bank)
    try:
        luna.predict(document, entities, candidates, luna_context)
    finally:
        luna.close()
    luna_scores = {(s["source"], s["target"]): s["positive_probability"]
                   for s in luna_context.prediction_metadata["decision_scores"]}
    decision_context = Journal(root / "receipts" / (doc + "-d1.jsonl"), doc, "d1")
    decision = JevDecisionPredictor(dc, max_workers=cfg["decision_workers"])
    try:
        decision.predict(document, entities, candidates, decision_context)
    finally:
        decision.close()
    d1_scores = {(s["source"], s["target"]): s["positive_probability"]
                 for s in decision_context.prediction_metadata["decision_scores"]}
    scores = [{"a": a, "b": b, "luna": luna_scores[(a, b)], "d1": d1_scores[(a, b)]} for a, b in candidates]
    atomic_json(path, {"signature": signature, "scores": scores,
                       "resources": {"luna": luna_context.resources(), "d1": decision_context.resources()}})
    emit("CONFIRM_BASELINE_DONE", {"doc": doc, "pairs": len(scores),
                                 "luna": luna_context.resources(), "d1": decision_context.resources()})
    return doc, scores


def produce(cfg, data, bank, root, signature):
    started = budget_status(cfg)["accounted_or_reserved_usd"]
    rows = run_parallel([lambda d=d: baseline_one(cfg, data, bank, root, signature, d) for d in data.docs],
                        cfg["document_workers"])
    for doc, scores in rows:
        data.set_baseline(doc, scores)
    band = {d: band_verifier.band_pairs(data, d, *cfg["band"]) for d in data.docs}
    emit("CONFIRM_BAND", {"unordered_pairs": sum(map(len, band.values())),
                          "documents_with_band": sum(bool(v) for v in band.values())})
    client, spend = client_from(cfg, band_verifier.parse_available), Spend()

    def judge_one(doc):
        path = root / "judge" / (doc + ".json")
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["signature"] != signature:
                raise ValueError("incompatible judge cache")
            return doc, saved["probabilities"]
        votes = {}
        for index, part in enumerate(chunks(band[doc], cfg["pairs_per_call"])):
            for sample in range(1, cfg["samples"] + 1):
                pairs = list(part)
                random.Random(f"{doc}::{sample}").shuffle(pairs)
                probs, _, repairs = band_verifier.ask_complete(client, data, doc, pairs, sample, spend)
                for key, p in probs.items():
                    votes.setdefault(key, []).append(p)
                emit("CONFIRM_JUDGE_DONE", {"doc": doc, "chunk": index, "sample": sample,
                                           "pairs": len(part), "repairs": repairs})
        if any(len(v) != cfg["samples"] for v in votes.values()):
            raise ValueError("judge samples are incomplete")
        probs = {k: float(np.mean(v)) for k, v in votes.items()}
        if {tuple(k.split(",")) for k in probs} != set(band[doc]):
            raise ValueError("judge band coverage differs")
        atomic_json(path, {"signature": signature, "probabilities": probs})
        return doc, probs

    try:
        judged = run_parallel([lambda d=d: judge_one(d) for d in data.docs], cfg["judge_workers"])
    finally:
        client.close()
    result = {"schema": "workere-confirmation-inference/v1", "signature": signature,
              "public_fingerprint": data.public_fingerprint, "baseline": dict(rows), "judge": dict(judged),
              "judge_resources": spend.value,
              "producer_additional_accounted_usd": budget_status(cfg)["accounted_or_reserved_usd"] - started}
    atomic_json(root / "complete.json", result)
    return result


def shared_inference(cfg, data, bank):
    signature = shared_signature(cfg, data)
    root = Path(cfg["work_root"]).expanduser() / signature
    root.mkdir(parents=True, exist_ok=True)
    with (root / "producer.lock").open("a") as lock:
        last = 0.
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() - last >= 30:
                    emit("CONFIRM_SHARED_WAIT", {"variant": cfg["variant"], "root": str(root)})
                    last = time.monotonic()
                time.sleep(2)
        path = root / "complete.json"
        if path.exists():
            result = json.loads(path.read_text())
            if result["signature"] != signature:
                raise ValueError("completed inference signature differs")
            emit("CONFIRM_SHARED_REUSED", {"signature": signature, "additional_api_calls": 0})
            return result, True
        emit("CONFIRM_SHARED_PRODUCER", {"variant": cfg["variant"], "signature": signature})
        return produce(cfg, data, bank, root, signature), False


def metrics(counts):
    total = np.sum(list(counts.values()), axis=0)
    tp, fp, fn = [int(x) for x in total]
    return {"f1": float(100 * f1_arr(total)), "tp": tp, "fp": fp, "fn": fn,
            "precision": 100 * tp / (tp + fp) if tp + fp else 0.,
            "recall": 100 * tp / (tp + fn) if tp + fn else 0.}


def contrast(per, base, docs, topics, samples=4000):
    candidate, control = metrics(per), metrics(base)
    out = {"candidate": candidate, "control": control, "delta_points": candidate["f1"] - control["f1"]}
    for unit in ("document", "topic"):
        groups = [[d] for d in docs] if unit == "document" else [
            [d for d in docs if topics[d] == topic] for topic in sorted(set(topics.values()), key=int)]
        p = np.array([np.sum([per[d] for d in group], axis=0) for group in groups])
        b = np.array([np.sum([base[d] for d in group], axis=0) for group in groups])
        draws = np.random.default_rng(42).integers(0, len(groups), size=(samples, len(groups)))
        delta = np.sort(100 * (f1_arr(p[draws].sum(1)) - f1_arr(b[draws].sum(1))))
        out[unit + "_bootstrap"] = {"samples": samples,
            "ci95": [float(delta[int(.025 * samples)]), float(delta[int(.975 * samples) - 1])],
            "family_ci97_5": [float(delta[int(.0125 * samples)]), float(delta[int(.9875 * samples) - 1])]}
    out["confirmation_criterion_met"] = (
        out["delta_points"] >= 1 and out["document_bootstrap"]["family_ci97_5"][0] > 0
        and out["topic_bootstrap"]["family_ci97_5"][0] > 0)
    return out


def score(cfg, data, freeze, inference):
    for doc, rows in inference["baseline"].items():
        data.set_baseline(doc, rows)
    judge = {d: {tuple(k.split(",")): p for k, p in m.items()} for d, m in inference["judge"].items()}
    for d in data.docs:
        if set(judge[d]) != set(band_verifier.band_pairs(data, d, *cfg["band"])):
            raise ValueError("final band inventory differs")
    rule = freeze["rules"][cfg["variant"]]
    # Predictions are finalized before the evaluator is allowed to read target labels.
    candidate_scores = {d: rule_scores(data, judge, rule, d) for d in data.docs}
    data.attach_gold(cfg["evaluation_only"])
    scorer = Scorer(data)
    base = scorer.fixed(lambda d: data.fused[d])
    per = scorer.fixed(lambda d: candidate_scores[d], rule["threshold"])
    p = {d: [per[d][k] for k in ("tp", "fp", "fn")] for d in data.docs}
    b = {d: [base[d][k] for k in ("tp", "fp", "fn")] for d in data.docs}
    return {"schema": "workere-confirmation-result/v1", "variant": cfg["variant"],
            "documents": len(data.docs), "topics": len(set(data.topic.values())),
            "band_unordered_pairs": sum(len(v) for v in judge.values()),
            "rule": rule, "rules_fingerprint": freeze["fingerprint"],
            "public_fingerprint": data.public_fingerprint,
            "primary": contrast(p, b, data.docs, data.topic),
            "per_document": {d: {"base": base[d], "candidate": per[d]} for d in data.docs}}


def run(config_path, output):
    cfg = load(config_path)
    data, freeze, bank = checked_inputs(cfg)
    preflight(cfg)
    emit("CONFIRM_PLAN", plan(cfg))
    inference, reused = shared_inference(cfg, data, bank)
    result = score(cfg, data, freeze, inference)
    result.update(shared_inference_reused=reused, effective_configuration=cfg,
                  inference_signature=inference["signature"],
                  inference_producer_accounted_usd=inference["producer_additional_accounted_usd"],
                  budget_status=budget_status(cfg))
    write_json(output, result)
    emit("CONFIRM_RESULT", result)
    return 0
