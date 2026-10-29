import copy
import contextlib
import io
import json
import pathlib
import tempfile
import unittest
from unittest.mock import patch

from pipelines.jev.n9 import (ARMS, prepare, validate_pairing, validate_content,
                              freeze_predictions, evaluate, export, account_costs)
from pipelines.jev.n9_transport import FreshLogicalBatchClient


class N9Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = prepare()

    def test_factorial_exact_pairing_and_common_admissible_pool(self):
        rows = self.rows
        self.assertEqual(len(rows), 240)
        self.assertEqual(len({r["custom_id"] for r in rows}), 240)
        validate_pairing(rows)
        for doc in {r["doc"] for r in rows}:
            pool = [r["eligible_demo_ids"] for r in rows if r["doc"] == doc]
            self.assertTrue(all(p == pool[0] for p in pool))
        changed = copy.deepcopy(rows)
        paired = next(r for r in changed if r["arm"] == "R0N1")
        paired["body"]["messages"][1]["content"] += " different document"
        with self.assertRaises(AssertionError):
            validate_pairing(changed)

    def test_same_bodies_are_separate_paid_slots_in_one_mixed_batch(self):
        doc = self.rows[0]["doc"]
        rows = [r for r in self.rows if r["doc"] == doc]
        self.assertEqual(len(rows), 24)
        self.assertLess(len({r["body_sha256"] for r in rows}), 24)
        posts, events = [], []

        def fake_http(client, method, url, payload=None):
            self.assertEqual(method, "POST")
            posts.append(payload)
            return {"id": "mock-document-batch", "status": "completed", "usage": {"cost": 0},
                    "results": [{"custom_id": r["custom_id"], "response": {"status_code": 200,
                        "body": {"id": "served-"+str(i), "model": "openai/gpt-6-luna",
                                 "provider": "OpenAI", "choices": [{"finish_reason": "stop",
                                  "message": {"content": "[]"}}]}}}
                                for i, r in enumerate(payload["requests"])]}

        with tempfile.TemporaryDirectory() as tmp, patch.object(FreshLogicalBatchClient, "_http", fake_http):
            client = FreshLogicalBatchClient(model="openai/gpt-6-luna:batch", model_version="gpt-6-luna-20260922",
                mode="live", cache_path=tmp+"/unused.sqlite", retries=0, max_calls=24,
                budget_usd=.61, max_cost_per_call_usd=.025, input_price_per_million=.1,
                max_context_tokens=65536, max_output_tokens=32768, reasoning_effort="high")
            client.decide_logical(rows, events.append)
            with self.assertRaises(RuntimeError):
                client.decide_logical(rows, events.append)
            client.close()
        self.assertEqual(len(posts), 1)
        self.assertEqual(len(posts[0]["requests"]), 24)
        self.assertEqual(len(events), 24)
        self.assertTrue(all(e["api_call"] and not e["cache_hit"] for e in events))

    def test_global_gate_blocks_truncation_reused_response_or_wrong_batch(self):
        events = []
        for row in self.rows:
            events.append({"logical_id": row["logical_id"], "custom_id": row["custom_id"],
                "request": row["body"], "api_call": True, "cache_hit": False, "error": None,
                "batch_id": "batch-"+row["doc"], "response": {"id": row["custom_id"],
                "model": "openai/gpt-6-luna", "provider": "OpenAI", "choices": [
                {"finish_reason": "stop", "message": {"content": "[]"}}]}})
        predictions, failures = freeze_predictions(self.rows, events)
        self.assertEqual(len(predictions), 80)
        self.assertFalse(failures)
        for change in ("truncation", "duplicate", "batch", "cache"):
            bad = copy.deepcopy(events)
            if change == "truncation":
                bad[-1]["response"]["choices"][0]["finish_reason"] = "length"
            elif change == "duplicate":
                bad[-1]["response"]["id"] = bad[0]["response"]["id"]
            elif change == "batch":
                bad[-1]["batch_id"] = bad[0]["batch_id"]
            else:
                bad[-1]["api_call"] = False
            predictions, failures = freeze_predictions(self.rows, bad)
            self.assertIsNone(predictions, change)
            self.assertTrue(failures, change)
        self.assertIsNone(freeze_predictions(self.rows, events[:-1])[0])

    def test_unknown_ids_and_last_label_rule(self):
        response = {"model": "openai/gpt-6-luna", "provider": "OpenAI", "choices": [
            {"finish_reason": "stop", "message": {"content":
             '[{"pair":"1,2","label":"causes"},{"pair":"1,2","label":"norel"}]'}}]}
        self.assertEqual(validate_content(response, {("1", "2")}), {("1", "2"): "norel"})
        with self.assertRaises(AssertionError):
            validate_content(response, {("8", "9")})

    def test_none_response_is_recorded_and_cannot_pass_gate(self):
        rows = [r for r in self.rows if r["doc"] == self.rows[0]["doc"]]
        events = []

        def fake_http(client, method, url, payload=None):
            return {"id": "none-body-batch", "status": "completed", "usage": {"cost": .01},
                    "results": [{"custom_id": r["custom_id"],
                                 "response": {"status_code": 200, "body": None}}
                                for r in payload["requests"]]}

        with tempfile.TemporaryDirectory() as tmp, patch.object(FreshLogicalBatchClient, "_http", fake_http):
            client = FreshLogicalBatchClient(model="openai/gpt-6-luna:batch", model_version="gpt-6-luna-20260922",
                mode="live", cache_path=tmp+"/unused.sqlite", retries=0, max_calls=24,
                budget_usd=.61, max_cost_per_call_usd=.025, input_price_per_million=.1,
                max_context_tokens=65536, max_output_tokens=32768, reasoning_effort="high")
            client.decide_logical(rows, events.append)
            client.close()
            with contextlib.redirect_stdout(io.StringIO()) as stream:
                export(pathlib.Path(tmp))
            self.assertIn("N9_EVIDENCE_END", stream.getvalue())
            self.assertTrue((pathlib.Path(tmp)/"luna-batch-jobs/final-retrieval.json").exists())
        self.assertEqual(len(events), 24)
        self.assertTrue(all(e["response"] is None and e["error"] for e in events))
        self.assertIsNone(freeze_predictions(self.rows, events)[0])

    def test_ambiguous_post_keeps_receipt_and_full_conservative_cost(self):
        rows = [r for r in self.rows if r["doc"] == self.rows[0]["doc"]]
        calls = []

        def uncertain_http(client, method, url, payload=None):
            calls.append(method)
            raise TimeoutError("submission accepted or lost; simulated ambiguity")

        with tempfile.TemporaryDirectory() as tmp, patch.object(FreshLogicalBatchClient, "_http", uncertain_http):
            root = pathlib.Path(tmp)
            client = FreshLogicalBatchClient(model="openai/gpt-6-luna:batch", model_version="gpt-6-luna-20260922",
                mode="live", cache_path=str(root/"batches"/rows[0]["doc"]/"unused.sqlite"), retries=0,
                max_calls=24, budget_usd=.61, max_cost_per_call_usd=.025, input_price_per_million=.1,
                max_context_tokens=65536, max_output_tokens=32768, reasoning_effort="high")
            for _ in range(2):
                with self.assertRaises(RuntimeError):
                    client.decide_logical(rows, lambda e: self.fail("no response expected"))
            client.close()
            receipt = json.loads((client.batch_dir/"one-shot.json").read_text())
            self.assertEqual(receipt["status"], "submitting")
            self.assertEqual(calls, ["POST"])
            self.assertEqual(account_costs(root)["conservative_usd"], 6)
            with contextlib.redirect_stdout(io.StringIO()):
                export(root)
            manifest = json.loads((root/"manifest.json").read_text())
            self.assertIn("batches/"+rows[0]["doc"]+"/luna-batch-jobs/submission-payload.json", manifest["files_sha256"])

    def test_terminal_partial_batch_preserves_raw_and_no_partial_scores(self):
        rows = [r for r in self.rows if r["doc"] == self.rows[0]["doc"]]

        def terminal_http(client, method, url, payload=None):
            return {"id": "incomplete-batch", "status": "failed", "results": [], "usage": {}}

        with tempfile.TemporaryDirectory() as tmp, patch.object(FreshLogicalBatchClient, "_http", terminal_http):
            root = pathlib.Path(tmp)
            client = FreshLogicalBatchClient(model="openai/gpt-6-luna:batch", model_version="gpt-6-luna-20260922",
                mode="live", cache_path=str(root/"batches"/rows[0]["doc"]/"unused.sqlite"), retries=0,
                max_calls=24, budget_usd=.61, max_cost_per_call_usd=.025, input_price_per_million=.1,
                max_context_tokens=65536, max_output_tokens=32768, reasoning_effort="high")
            with self.assertRaises(ValueError):
                client.decide_logical(rows, lambda e: self.fail("no complete batch"))
            client.close()
            raw = json.loads((client.batch_dir/"final-retrieval.json").read_text())
            self.assertEqual(raw["status"], "failed")
            self.assertEqual(account_costs(root)["conservative_usd"], 6)
            self.assertIsNone(freeze_predictions(self.rows, [])[0])

    def test_interaction_and_adjusted_bootstrap_use_condition_micro_f1(self):
        # Two balanced topics, one positive and one negative pair each direction.
        controls = []
        decisions = []
        for topic in ("a", "b"):
            doc = topic+"_synthetic.xml"
            for pair, gold in ((("1", "2"), True), (("2", "1"), True),
                               (("1", "3"), False), (("3", "1"), False)):
                controls.append({"doc": doc, "a": pair[0], "b": pair[1], "gold": gold})
                for schedule in (0, 1):
                    for arm in ARMS:
                        pred = gold and (arm != "R0N0" or topic == "a")
                        if arm == "R1N1" and not gold:
                            pred = True
                        decisions.append({"schedule": schedule, "arm": arm, "doc": doc,
                                          "a": pair[0], "b": pair[1], "positive": pred})
        result = evaluate(decisions, controls, draws=120)
        vals = result["systems"]["0"]
        self.assertAlmostEqual(vals["R0N0"]["F1"], 200/3)
        self.assertEqual(vals["R0N1"]["F1"], 100)
        self.assertAlmostEqual(result["contrasts"]["interaction"]["mean_delta_points"], -200/3)
        self.assertEqual(result["bootstrap"]["percentile_indices_zero_based"], [0, 119])


if __name__ == "__main__":
    unittest.main()
