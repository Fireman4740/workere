"""New-signal experiments: request isolation, cache identity, selection and precision gate."""
import copy
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipelines.judges import band_verifier as band, coref_agreement as coref
from pipelines.judges.common import (
    MODEL, DeepSeekClient, Dev44, Scorer, components, last_json, link_quality,
)
from pipelines.judges.driver import load, plan

ROOT = Path(__file__).resolve().parents[1]


def tiny_data():
    obj = object.__new__(Dev44)
    obj.docs = ["1_x", "2_x"]
    obj.topic = {"1_x": "1", "2_x": "2"}
    obj.suite, obj.by_doc, obj.keys, obj.fused = {}, {}, {}, {}
    for d in obj.docs:
        obj.suite[d] = {"tokens": ["storm", "flood", "storm"], "spans": [[0], [1], [2]],
                        "mentions": ["1", "2", "3"], "surfaces": {"1": "storm", "2": "flood", "3": "storm"}}
        probs = {("1", "2"): .4, ("2", "3"): .3 if d == "1_x" else .6, ("1", "3"): .1}
        obj.by_doc[d], obj.fused[d] = {}, {}
        for pair, p in probs.items():
            for key in (pair, pair[::-1]):
                obj.by_doc[d][key] = {"gold": pair != ("1", "3") if d == "1_x" else pair == ("2", "3")}
                obj.fused[d][key] = p
        obj.keys[d] = list(obj.by_doc[d])
    return obj


class Validation(unittest.TestCase):
    def test_probabilities_require_exact_coverage_and_numeric_range(self):
        body = {"messages": [{"content": "Pairs to judge:\n- 1,2 (a / b)\n"}]}
        self.assertEqual(band.parse_probabilities('[{"pair":"2,1","p":0.3}]', body), {"1,2": .3})
        for answer in ("[]", '[{"pair":"1,2","p":true}]', '[{"pair":"1,2","p":NaN}]',
                       '[{"pair":"1,2","p":1.2}]', '[{"pair":"1,3","p":0.5}]',
                       '[{"pair":"1,2","p":0.5},{"pair":"2,1","p":0.5}]'):
            with self.subTest(answer=answer), self.assertRaises(ValueError):
                band.parse_probabilities(answer, body)

    def test_coreference_judgements_are_complete_booleans(self):
        body = {"messages": [{"content": "Pairs to verify:\n- 1,2 (a / b)\n"}]}
        for answer in ("[]", '[{"pair":"1,2","same":1}]',
                       '[{"pair":"1,2","same":true},{"pair":"1,2","same":true}]'):
            with self.subTest(answer=answer), self.assertRaises(ValueError):
                coref.parse_judgements(answer, body)
        self.assertEqual(coref.parse_judgements('[{"pair":"2,1","same":false}]', body), {"1,2": False})

    def test_json_parser_and_discontinuous_mentions(self):
        self.assertEqual(last_json('Reason ["x"] then [{"pair":"1,2","note":"]"}]', list),
                         [{"pair": "1,2", "note": "]"}])
        data = tiny_data()
        data.suite["1_x"]["spans"] = [[0, 2], [1]]
        data.suite["1_x"]["mentions"] = ["1", "2"]
        self.assertEqual(data.marked_text("1_x"), "<1 storm> <2 flood> <1 storm>")

    def test_live_plan_matches_frozen_population(self):
        cfg = load(ROOT / "experiments/jev/judges/j2.yaml")
        planned = plan(cfg)
        self.assertEqual((planned["documents"], planned["band_unordered_pairs"]), (44, 1089))
        data = Dev44(cfg["controls"], cfg["suites_dir"])
        self.assertAlmostEqual(100 * sum(2 * c["tp"] for c in Scorer(data).fixed(lambda d: data.fused[d]).values())
                               / sum(2 * c["tp"] + c["fp"] + c["fn"]
                                     for c in Scorer(data).fixed(lambda d: data.fused[d]).values()),
                               47.32, places=2)

    def test_effort_is_sent_and_changes_cache_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            seen = []
            def transport(body):
                seen.append(body)
                return {"model": MODEL, "choices": [{"finish_reason": "stop",
                         "message": {"content": '[{"pair":"1,2","p":0.6}]'}}],
                        "usage": {"cost": .001, "prompt_tokens": 20, "completion_tokens": 30,
                                  "completion_tokens_details": {"reasoning_tokens": 20}}}
            params = dict(parse=band.parse_probabilities, model=MODEL, model_version="v1", mode="live",
                          cache_path=str(Path(tmp) / "cache.sqlite"), budget_usd=1, max_calls=5,
                          max_output_tokens=65536, max_context_tokens=1048576,
                          input_price_per_million=.3, output_price_per_million=1.2,
                          max_cost_per_call_usd=.1, transport=transport, requests_per_minute=1000000)
            for effort in ("high", "max", "max"):
                with DeepSeekClient(**params, reasoning_effort=effort) as client:
                    reply = client.ask("no labels", "Pairs to judge:\n- 1,2 (a / b)\n", 1)
            self.assertEqual(len(seen), 2)
            self.assertEqual([b["reasoning"]["effort"] for b in seen], ["high", "max"])
            self.assertTrue(seen[-1]["provider"]["require_parameters"])
            self.assertTrue(reply["cache_hit"])

    def test_luna_cache_is_read_only_and_inventory_is_checked(self):
        data = tiny_data()
        with tempfile.TemporaryDirectory() as tmp:
            for d in data.docs:
                db = sqlite3.connect(Path(tmp) / f"r3-t2-coref-{d}.sqlite")
                db.execute("CREATE TABLE decisions_v1 (key TEXT, request TEXT, response TEXT)")
                req = {"model": "openai/gpt-6-luna:batch", "messages": [
                    {"content": f"Document {d}:\n{data.marked_text(d)}"}]}
                rep = {"model": "openai/gpt-6-luna", "choices": [
                    {"finish_reason": "stop", "message": {"content": '[["1","3"]]'}}]}
                db.execute("INSERT INTO decisions_v1 VALUES (?,?,?)", ("k", json.dumps(req), json.dumps(rep)))
                db.commit()
                db.close()
            self.assertEqual(coref.luna_groups(data, tmp)["1_x"], [["1", "3"]])
            data.suite["1_x"]["mentions"].append("4")
            with self.assertRaisesRegex(ValueError, "inventory"):
                coref.luna_groups(data, tmp)


class Protocol(unittest.TestCase):
    def test_lto_never_selects_on_held_out_topic_labels(self):
        data = tiny_data()
        variants = {"base": lambda d: data.fused[d],
                    "higher": lambda d: {k: min(1, s + .3) for k, s in data.fused[d].items()}}
        first = Scorer(data).lto(variants, thresholds=.39)[1]["1"]
        for row in data.by_doc["1_x"].values():
            row["gold"] = not row["gold"]
        second = Scorer(data).lto(variants, thresholds=.39)[1]["1"]
        self.assertEqual(first, second)

    def test_average_precision_treats_ties_as_one_threshold(self):
        data = tiny_data()
        expected = sum(r["gold"] for d in data.docs for r in data.by_doc[d].values()) / 12
        self.assertAlmostEqual(Scorer(data).ap(lambda d: {k: .5 for k in data.keys[d]}), expected)

    def test_transitive_closure_counts_unverified_links_for_the_gate(self):
        closed = components([("1", "2"), ("2", "3")], ["1", "2", "3"])
        quality = link_quality({"d": closed}, {"d": [["1", "2"]]}, ["d"])
        self.assertEqual(quality["predicted_links"], 3)
        self.assertAlmostEqual(quality["precision"], 1 / 3)

    def test_failed_precision_gate_stops_before_causal_scoring(self):
        data = tiny_data()
        cfg = {"dataset": {"repo_id": "unused", "revision": "pinned"}, "gate_precision": .85}
        luna = {d: [["1", "3"]] for d in data.docs}
        agreed = {d: [["1", "2"]] for d in data.docs}
        with patch.object(coref, "gold_chains", return_value=luna), \
                patch.object(coref, "Scorer", side_effect=AssertionError("must not evaluate causality")):
            out = coref.score(cfg, data, luna, agreed, {})
        self.assertFalse(out["gate_passed"])
        self.assertEqual(out["causal_evaluation"], "skipped_intrinsic_gate_failed")

    def test_end_to_end_band_and_coref_without_gold_in_requests(self):
        data = tiny_data()
        requests = []
        class Scripted:
            def ask(self, system, user, sample):
                requests.append((system, user, sample))
                is_coref = "Pairs to verify:" in user
                body = {"messages": [{"content": user}]}
                wanted = coref.requested_pairs(body) if is_coref else band._requested(body)
                answer = [{"pair": k, **({"same": True} if is_coref else {"p": .6})} for k in sorted(wanted)]
                return {"content": json.dumps(answer), "cache_hit": False, "cost_usd": 0,
                        "api_attempts": 0, "usage": {}}
            def close(self):
                pass
        cfg = {"controls": "", "suites_dir": "", "samples": 2, "workers": 2, "pairs_per_call": 1,
               "weights": [0, .5, 1], "band": [.25, .55]}
        with patch.object(band, "Dev44", return_value=data), \
                patch.object(band, "client_from", return_value=Scripted()):
            result = band.run(cfg)
        self.assertEqual(result["band_pairs"], 3)
        self.assertEqual(result["spend"]["calls"], 6)
        for d, probs in result["judge_probabilities"].items():
            for key in probs:
                a, b = key.split(",")
                stacked = band.stacked(data, {d2: {tuple(k.split(",")): p for k, p in m.items()}
                                             for d2, m in result["judge_probabilities"].items()}, .5)
                self.assertEqual(stacked(d)[(a, b)], stacked(d)[(b, a)])
        first_requests = copy.deepcopy(requests)
        requests.clear()
        for d in data.docs:
            for row in data.by_doc[d].values():
                row["gold"] = not row["gold"]
        with patch.object(band, "Dev44", return_value=data), \
                patch.object(band, "client_from", return_value=Scripted()):
            band.run(cfg)
        self.assertEqual(sorted(requests), sorted(first_requests))
        luna = {d: [["1", "3"]] for d in data.docs}
        cfg.update(luna_cache_dir="unused", gate_precision=.85, dataset={"repo_id": "", "revision": ""})
        with patch.object(coref, "Dev44", return_value=data), \
                patch.object(coref, "luna_groups", return_value=luna), \
                patch.object(coref, "gold_chains", return_value=luna), \
                patch.object(coref, "client_from", return_value=Scripted()):
            out = coref.run(cfg)
        self.assertTrue(out["gate_passed"])
        self.assertEqual(out["spend"]["calls"], 2)
        for system, user, _ in requests:
            self.assertNotIn('"gold"', system + user)
            self.assertNotIn('"luna"', system + user)
            self.assertNotIn('"d1"', system + user)


if __name__ == "__main__":
    unittest.main()
