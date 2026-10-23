"""Frozen-rule confirmation: separation, request parity and one shared inference producer."""
import copy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from benchmark.io import fingerprint
from benchmark.schema import Document, Entity
from pipelines.judges import confirmation as conf
from pipelines.judges import band_verifier as band
from pipelines.judges.calibration import load_freeze, logistic_prob, rule_scores
from pipelines.jev.llm import LunaDocumentPredictor

ROOT = Path(__file__).resolve().parents[1]


def fixture(tmp):
    tmp = Path(tmp)
    data = {"schema": "workere-confirmation-public/v1", "revision": "pinned", "documents": []}
    for d in ("101_doc", "102_doc"):
        data["documents"].append({"id": d, "tokens": ["storm", "caused", "flood"],
            "sentences": [[0, 3]], "lang": "eng", "mentions": ["1", "2"], "spans": [[0], [2]],
            "surfaces": {"1": "storm", "2": "flood"}, "sentence_of": {"1": 0, "2": 0},
            "pair_list": [["1", "2"], ["2", "1"]]})
    data["fingerprint"] = fingerprint(data)
    (tmp / "public.json").write_text(json.dumps(data))
    frozen = json.loads((ROOT / "experiments/jev/confirmation/frozen-rules.json").read_text())
    (tmp / "rules.json").write_text(json.dumps(frozen))
    selection = {"frozen_rules_fingerprint": frozen["fingerprint"], "documents": [{"id": d["id"]} for d in data["documents"]]}
    (tmp / "selection.json").write_text(json.dumps(selection))
    (tmp / "bank.json").write_text(json.dumps({"documents": []}))
    (tmp / "gold.json").write_text(json.dumps({"public_fingerprint": data["fingerprint"],
        "positive_pairs": {"101_doc": [["1", "2"], ["2", "1"]], "102_doc": []}}))
    cfg = conf.load(ROOT / "experiments/jev/confirmation/config.yaml")
    cfg.update(public_inputs=str(tmp / "public.json"), evaluation_only=str(tmp / "gold.json"),
               frozen_rules=str(tmp / "rules.json"), selection=str(tmp / "selection.json"),
               demo_bank=str(tmp / "bank.json"), work_root=str(tmp / "shared"))
    cfg["budget"] = {"path": str(tmp / "budget.sqlite"), "scope": "shared", "limit_usd": 3, "initial_spend_usd": 0}
    return cfg


class Confirmation(unittest.TestCase):
    def test_population_is_reserved_and_plan_does_not_read_target_labels(self):
        cfg = conf.load(ROOT / "experiments/jev/confirmation/config.yaml")
        reserve = {d["id"] for d in json.loads((ROOT / "experiments/jev/round3/round3-selection.json").read_text())["holdout_rest"]}
        data, frozen, _ = conf.checked_inputs(cfg)
        self.assertEqual((len(data.docs), len(set(data.topic.values()))), (30, 20))
        self.assertEqual(sum(map(len, data.keys.values())), 21014)
        self.assertTrue(set(data.docs) <= reserve)
        self.assertFalse(set(data.docs) & set(frozen["training_documents"]))
        with patch.object(conf.Cohort, "attach_gold", side_effect=AssertionError("labels read during plan")):
            self.assertFalse(conf.plan(cfg)["annotations_read"])
        self.assertTrue(all("gold" not in r for v in data.by_doc.values() for r in v.values()))

    def test_calibration_freeze_detects_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = fixture(tmp)
            p = Path(cfg["frozen_rules"])
            frozen = json.loads(p.read_text())
            frozen["rules"]["mean"]["threshold"] = .1
            p.write_text(json.dumps(frozen))
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                load_freeze(p)

    def test_public_payload_rejects_annotations(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = fixture(tmp)
            p = Path(cfg["public_inputs"]);data = json.loads(p.read_text())
            data["documents"][0]["gold"] = []
            data["fingerprint"] = fingerprint({k: v for k, v in data.items() if k != "fingerprint"})
            p.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "annotation"):
                conf.Cohort(p)

    def test_mean_matches_frozen_j2_formula_and_logistic_keeps_outside_decisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = fixture(tmp);data, frozen, _ = conf.checked_inputs(cfg)
            for doc in data.docs:
                data.set_baseline(doc, [{"a": a, "b": b, "luna": .5, "d1": .3} for a, b in data.keys[doc]])
            judge = {d: {("1", "2"): .8} for d in data.docs}
            values = rule_scores(data, judge, frozen["rules"]["mean"], data.docs[0])
            self.assertTrue(all(abs(v - .6) < 1e-12 for v in values.values()))
            p = logistic_prob(frozen["rules"]["logistic"], [.4], [.8])
            self.assertTrue(np.isfinite(p).all() and (p > 0).all() and (p < 1).all())
            judge = {d: {} for d in data.docs}
            # No judged pair means the logistic branch preserves the base decision.
            values = rule_scores(data, judge, frozen["rules"]["logistic"], data.docs[0])
            self.assertTrue(all(v == 1 for v in values.values()))

    def test_luna_requests_match_reference_with_the_frozen_bank(self):
        cfg = conf.load(ROOT / "experiments/jev/confirmation/config.yaml")
        data, _, bank = conf.checked_inputs(cfg)
        source = data.suite[data.docs[0]]
        document = Document(source["id"], source["tokens"], source["sentences"], source["lang"])
        entities = tuple(Entity(m, s) for m, s in zip(source["mentions"], source["spans"]))
        requests = []
        class ScriptedBatch:
            mode = "live"
            identity = {"model": "openai/gpt-6-luna:batch"}
            def decide_many(self, messages, callbacks):
                requests.append(copy.deepcopy(messages))
                return [{"response": {"model": "openai/gpt-6-luna", "choices": [{"message": {"content": "[]"}}]}}
                        for _ in messages]
        def predictor(cls):
            p = object.__new__(cls)
            p.client, p.prompt, p.bank, p.passes = ScriptedBatch(), conf.read_yaml(cfg["luna_prompt"]), bank, 7
            return p
        for cls in (LunaDocumentPredictor, conf.FrozenLuna):
            class Context:
                seed = 42
                rng = random.Random(42)
                cost_usd = 0
                prediction_metadata = {}
                def record_usage(self, **event): pass
            predictor(cls).predict(document, entities, tuple(data.keys[document.id]), Context())
        self.assertEqual(requests[0], requests[1])
        self.assertEqual(len(requests[0]), 7)
        self.assertTrue(all(len(m) == 4 for m in requests[0]))  # system, demo question/answer, target
        demo_ids = {d["id"] for d in bank}
        self.assertNotIn(document.id, demo_ids)

    def test_two_nodes_share_one_producer_and_target_gold_is_read_only_for_scoring(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = fixture(tmp);cfg2 = copy.deepcopy(cfg);cfg2["variant"] = "logistic"
            data, frozen, bank = conf.checked_inputs(cfg)
            data2, frozen2, bank2 = conf.checked_inputs(cfg2)
            self.assertEqual(conf.shared_signature(cfg, data), conf.shared_signature(cfg2, data2))
            calls = []
            class FakeLuna:
                def __init__(self, *args): pass
                def predict(self, document, entities, candidates, context):
                    calls.append(("luna", document.id))
                    time.sleep(.02)
                    context.prediction_metadata["decision_scores"] = [
                        {"source": a, "target": b, "positive_probability": .5} for a, b in candidates]
                def close(self): pass
            class FakeDecision:
                def __init__(self, *args, **kwargs): pass
                def predict(self, document, entities, candidates, context):
                    calls.append(("d1", document.id))
                    context.prediction_metadata["decision_scores"] = [
                        {"source": a, "target": b, "positive_probability": .3} for a, b in candidates]
                def close(self): pass
            class FakeJudge:
                def ask(self, system, user, sample):
                    self_body = {"messages": [{"content": user}]}
                    pairs = band._requested(self_body)
                    calls.append(("judge", sample))
                    assert '"positive_pairs"' not in user and '"gold"' not in user
                    return {"content": json.dumps([{"pair": p, "p": .8} for p in pairs]),
                            "usage": {}, "cost_usd": 0, "cache_hit": False, "api_attempts": 0}
                def close(self): pass
            with patch.object(conf, "FrozenLuna", FakeLuna), patch.object(conf, "JevDecisionPredictor", FakeDecision), \
                    patch.object(conf, "client_from", return_value=FakeJudge()):
                with ThreadPoolExecutor(max_workers=2) as pool:
                    a = pool.submit(conf.shared_inference, cfg, data, bank)
                    b = pool.submit(conf.shared_inference, cfg2, data2, bank2)
                    first, second = a.result(), b.result()
            self.assertEqual(sorted([first[1], second[1]]), [False, True])
            self.assertEqual(sum(k == "luna" for k, _ in calls), 2)
            self.assertEqual(sum(k == "d1" for k, _ in calls), 2)
            self.assertEqual(sum(k == "judge" for k, _ in calls), 4)
            self.assertTrue(all("gold" not in row for values in data.by_doc.values() for row in values.values()))
            out = conf.score(cfg, data, frozen, first[0])
            out2 = conf.score(cfg2, data2, frozen2, second[0])
            self.assertEqual(out["documents"], 2)
            self.assertEqual(out["primary"]["control"], out2["primary"]["control"])
            self.assertIn("family_ci97_5", out["primary"]["document_bootstrap"])

    def test_incomplete_prediction_never_reaches_gold(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = fixture(tmp);data, frozen, _ = conf.checked_inputs(cfg)
            bad = {"baseline": {d: [] for d in data.docs}, "judge": {}}
            with patch.object(data, "attach_gold", side_effect=AssertionError("gold read before completeness")):
                with self.assertRaisesRegex(ValueError, "cover candidates"):
                    conf.score(cfg, data, frozen, bad)

    def test_identical_controls_cannot_pass_confirmation(self):
        counts = {"1_a": [3, 2, 1], "1_b": [1, 0, 2], "2_c": [2, 1, 1]}
        out = conf.contrast(counts, counts, list(counts), {d: d.split("_")[0] for d in counts})
        self.assertEqual(out["delta_points"], 0)
        self.assertEqual(out["topic_bootstrap"]["family_ci97_5"], [0, 0])
        self.assertFalse(out["confirmation_criterion_met"])


if __name__ == "__main__":
    unittest.main()
