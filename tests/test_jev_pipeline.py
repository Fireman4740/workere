"""Noul end-to-end and API safety tests. Mock results never stand in for model measurements."""
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from benchmark.configuration import load_experiment
from benchmark.engine import execute, reevaluate
from benchmark.io import read_json, read_jsonl, write_json
from benchmark.protocols import public_inputs
from benchmark.suite import freeze
from pipelines.jev.client import BudgetExceeded, DecisionClient
from pipelines.jev.postprocessor import noul_label
from pipelines.jev.state_builder import build_state, noul_question, questions_for
from tests.test_benchmark import config, rows
from tests.test_framework_protocols import component


class NoulV1(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.suite = freeze(config(), rows())

    def client(self, **overrides):
        options = {"model": "typesafe/jev-1.14", "model_version": "1.14", "mode": "mock",
                   "cache_path": str(self.root / "cache.sqlite"), "requests_per_minute": 100000,
                   "max_calls": 100}
        options.update(overrides)
        client = DecisionClient(**options)
        self.addCleanup(client.close)
        return client

    def test_state_has_only_selected_reference_mentions_and_no_gold_relations(self):
        inputs = next(public_inputs(self.suite, "relations_gold"))
        a, b = inputs.reference_entities[:2]
        state = build_state(inputs.document, a, b)
        self.assertIn('"id": "a"', state)
        self.assertIn('"token_indices": [0]', state)
        self.assertNotIn("<a rain>", state)
        self.assertNotIn("annots", state)
        self.assertNotIn("gold", state)
        self.assertNotIn('"label":', state)
        self.assertEqual(noul_label(.5), "causal")
        self.assertEqual(noul_label(.499), "norel")
        self.assertEqual(questions_for("choice", "Custom decision question")["relation"]["instructions"], "Custom decision question")
        self.assertEqual(questions_for("noul"), noul_question())

    def test_mock_cache_and_version_keys(self):
        client = self.client()
        first = client.decide("rain caused flooding", noul_question())
        second = client.decide("rain caused flooding", noul_question())
        self.assertFalse(first["cache_hit"])
        self.assertTrue(second["cache_hit"])
        self.assertEqual(first["response"], second["response"])
        self.assertEqual(client.calls, 1)
        different = self.client(model_version="1.13")
        self.assertNotEqual(different.decide("rain caused flooding", noul_question())["cache_key"], first["cache_key"])

    def test_paid_budget_is_enforced_before_transport_and_context_is_not_truncated(self):
        calls = []
        client = self.client(mode="live", budget_usd=0, transport=lambda body: calls.append(body))
        with self.assertRaises(BudgetExceeded):
            client.decide("rain caused flooding", noul_question())
        self.assertEqual(calls, [])
        self.assertFalse(self.client(max_cost_per_call_usd=.0000001).decide("mock has no charge", noul_question())["cache_hit"])
        short = self.client(max_context_tokens=10)
        with self.assertRaisesRegex(RuntimeError, "context_overflow"):
            short.decide("rain caused flooding", noul_question())

    def test_retry_journal_and_non_retryable_failure(self):
        import urllib.error
        from pipelines.jev.client import DecisionError
        events, calls = [], []
        def send(body):
            calls.append(body)
            if len(calls) == 1:
                raise urllib.error.HTTPError("https://example.invalid", 429, "rate limited", {}, None)
            return {"model": "test-served", "answers": {"causal": {"type": "noul", "noul": .7}}, "usage": {"cost": 0}}
        client = self.client(model="respan/span-01-lite", mode="live", transport=send)
        with patch("pipelines.jev.client.time.sleep"):
            packet = client.decide("Rain caused flooding", noul_question(), on_attempt=events.append)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["error"]["message"], "HTTP 429")
        self.assertEqual(packet["response"]["answers"]["causal"]["noul"], .7)
        self.assertTrue(client.decide("Rain caused flooding", noul_question())["cache_hit"])
        with self.assertRaises(BudgetExceeded):
            self.client(model="respan/span-01-lite", mode="live", transport=send, max_calls=0).decide("other", noul_question())
        bad = self.client(model="respan/span-01-lite", mode="live", retries=0,
            transport=lambda body: {"model": "test", "answers": {"causal": {"type": "noul", "noul": None}}})
        with self.assertRaises(DecisionError):
            bad.decide("missing probability", noul_question())

    def test_rate_slots_are_shared_between_clients_using_the_same_cache(self):
        a = self.client(model="respan/span-01-lite", mode="live", requests_per_minute=20)
        b = self.client(model="inception/mercury-decide:free", mode="live", requests_per_minute=20)
        with patch("pipelines.jev.client.time.time", return_value=1000):
            self.assertEqual(a._rate_slot(), 0)
            self.assertEqual(b._rate_slot(), 3)
            self.assertEqual(a._rate_slot(), 6)
        with patch("pipelines.jev.client.time.time", return_value=1010):
            self.assertEqual(b._rate_slot(), 0)

    def test_retry_after_is_honored_and_diagnostics_exclude_credentials(self):
        import urllib.error
        calls, events = [], []
        def send(body):
            calls.append(body)
            if len(calls) == 1:
                raise urllib.error.HTTPError("https://example.invalid", 429, "limited", {
                    "Retry-After": "120", "X-RateLimit-Remaining": "0",
                    "Authorization": "must-not-be-persisted"},
                    io.BytesIO(json.dumps({"error": {"metadata": {
                        "error_type": "rate_limit_exceeded", "raw": "must-not-be-persisted"}}}).encode()))
            return {"model": "served", "answers": {"causal": {"type": "noul", "noul": .7}}, "usage": {"cost": 0}}
        client = self.client(model="inception/mercury-decide:free", mode="live", transport=send)
        with patch.object(client, "_rate_slot", return_value=0), patch("pipelines.jev.client.time.sleep") as sleep:
            client.decide("unchanged public state", noul_question(), on_attempt=events.append)
        self.assertEqual(sum(c.args[0] for c in sleep.call_args_list), 120)
        self.assertEqual(events[0]["retry_delay_seconds"], 120)
        self.assertEqual(events[0]["error"]["http_status"], 429)
        self.assertEqual(events[0]["error"]["provider_metadata"]["error_type"], "rate_limit_exceeded")
        self.assertNotIn("must-not-be-persisted", json.dumps(events))
        # A repair's scheduling changes must reuse successful decisions unchanged.
        resumed = self.client(model="inception/mercury-decide:free", mode="live", max_calls=0,
                              retries=6, retry_backoff_seconds=60)
        self.assertTrue(resumed.decide("unchanged public state", noul_question())["cache_hit"])
        self.assertEqual(resumed.calls, 0)

    def test_retry_date_reset_and_long_cooldown_never_retry_early(self):
        import urllib.error
        from pipelines.jev.client import _server_wait, DecisionError
        self.assertEqual(_server_wait({"retry-after": "Thu, 01 Jan 1970 00:18:20 GMT"}, 1000), 100)
        self.assertEqual(_server_wait({"retry-after": "3", "x-ratelimit-reset": "1120"}, 1000), 120)
        self.assertEqual(_server_wait({"retry-after": "NaN", "x-ratelimit-reset": "garbage"}, 1000), 0)
        def send(body):
            raise urllib.error.HTTPError("https://example.invalid", 429, "limited", {"Retry-After": "300"}, None)
        client = self.client(model="inception/mercury-decide:free", mode="live", transport=send,
                              retry_max_wait_seconds=120)
        with patch.object(client, "_rate_slot", return_value=0), patch("pipelines.jev.client.time.sleep") as sleep:
            with self.assertRaisesRegex(DecisionError, "no early retry"):
                client.decide("blocked state", noul_question())
        self.assertEqual(client.calls, 1)
        sleep.assert_not_called()

    def test_429_cooldown_is_shared_and_401_is_not_retried(self):
        import urllib.error
        from pipelines.jev.client import DecisionError
        a = self.client(model="inception/mercury-decide:free", mode="live", requests_per_minute=20)
        b = self.client(model="respan/span-01-lite", mode="live", requests_per_minute=20)
        with patch("pipelines.jev.client.time.time", return_value=1000):
            a._defer_requests(120)
            self.assertEqual(b._rate_slot(), 120)
        def send(body):
            raise urllib.error.HTTPError("https://example.invalid", 401, "unauthorized", {}, None)
        c = self.client(model="inception/mercury-decide:free", mode="live", transport=send)
        with patch.object(c, "_rate_slot", return_value=0), patch("pipelines.jev.client.time.sleep") as sleep:
            with self.assertRaisesRegex(DecisionError, "HTTP 401"):
                c.decide("another state", noul_question())
        self.assertEqual(c.calls, 1)
        sleep.assert_not_called()

    def test_registered_postprocessor_reclassifies_both_ways_without_a_client(self):
        from benchmark.interfaces import Context
        from benchmark.schema import Prediction, Relation
        from pipelines.jev.postprocessor import NoulPostProcessor
        item = next(public_inputs(self.suite, "relations_gold"))
        a, b = item.reference_entities[:2]
        prediction = Prediction(item.document.id, item.reference_entities,
            (Relation(a.id, b.id, "norel", .6),), {"decision_type": "noul", "threshold": .5,
                "decision_scores": [{"source": a.id, "target": b.id, "positive_probability": .4}]})
        context = Context(42, "relations_gold", item.document.id)
        with patch.object(DecisionClient, "_send", side_effect=AssertionError("offline postprocessing")):
            positive = NoulPostProcessor(.3).process(item.document, prediction, context)
            negative = NoulPostProcessor(.8).process(item.document, positive, context)
        self.assertEqual(positive.relations[0].label, "causal")
        self.assertEqual(positive.relations[0].confidence, .4)
        self.assertEqual(negative.relations[0].label, "norel")
        self.assertEqual(negative.relations[0].confidence, .6)
        self.assertEqual(prediction.metadata["threshold"], .5)

    def test_noul_mock_runs_through_real_historical_scorer_and_replays_without_client(self):
        write_json(self.root / "suite.json", self.suite)
        cfg = {"schema": "workere-experiment/v1", "name": "jev-noul-mock", "seed": 42,
            "dataset": {"name": "synthetic", "split": "train", "suite": "suite.json"},
            "protocol": "relation_gold_mentions", "models": [{"id": "typesafe/jev-1.14", "version": "1.14"}],
            "plugins": ["pipelines.jev.pipeline"], "pipeline": {"entity_extractor": component("reference"),
                "candidate_generator": component("historical"), "relation_predictor": component("jev_decision",
                    client={"model": "typesafe/jev-1.14", "model_version": "1.14", "mode": "mock",
                            "cache_path": "cache.sqlite", "max_calls": 100})},
            "evaluation": {"bootstrap_samples": 20, "primary_metric": "binary"},
            "execution": {"allow_network": False, "allow_paid": False}, "output_dir": "runs"}
        source = self.root / "experiment.yaml"
        source.write_text(yaml.safe_dump(cfg))
        result = execute(load_experiment(source))
        directory = Path(result["run_dir"])
        self.assertEqual(result["status"], "completed")
        metrics = read_json(directory / "metrics.json")
        metadata = read_json(directory / "metadata.json")
        predictions = read_jsonl(directory / "predictions.jsonl")
        self.assertEqual(metadata["resources"]["execution_mode"], "mock")
        self.assertEqual(metadata["resources"]["api_cost_usd"], 0)
        self.assertEqual(len(predictions), 3)
        usage = read_jsonl(directory / "usage.jsonl")
        self.assertEqual(sum(row["calls"] for row in usage), 0)
        self.assertEqual(sum(row["decision_events"] for row in usage), sum(len(d["pair_list"]) for d in self.suite["documents"]))
        self.assertTrue((directory / "source.zip").is_file())
        self.assertTrue((directory / "environment.json").is_file())
        with patch("benchmark.engine.load_plugins", side_effect=AssertionError("must not load any model")):
            replay = reevaluate(directory, self.root / "replayed.json")
        self.assertEqual(replay, metrics)
        # Changing only the classification threshold reuses every pair's cached decision.
        cfg["pipeline"]["relation_predictor"]["params"]["threshold"] = .8
        source.write_text(yaml.safe_dump(cfg))
        second = execute(load_experiment(source))
        replay_usage = read_jsonl(Path(second["run_dir"]) / "usage.jsonl")
        self.assertTrue(all(e["cache_hit"] for row in replay_usage for e in row["provider_usage"]))


    def _mock_config(self, name, **params):
        if not (self.root / "suite.json").exists():
            write_json(self.root / "suite.json", self.suite)
        client = {"model": "typesafe/jev-1.14", "model_version": "1.14", "mode": "mock",
                  "cache_path": "cache.sqlite", "max_calls": 1000}
        return {"schema": "workere-experiment/v1", "name": name, "seed": 42,
            "dataset": {"name": "synthetic", "split": "train", "suite": "suite.json"},
            "protocol": "relation_gold_mentions", "models": [{"id": "typesafe/jev-1.14", "version": "1.14"}],
            "plugins": ["pipelines.jev.pipeline"], "pipeline": {"entity_extractor": component("reference"),
                "candidate_generator": component("historical"),
                "relation_predictor": component("jev_decision", client=client, **params)},
            "evaluation": {"bootstrap_samples": 20, "primary_metric": "binary"},
            "execution": {"allow_network": False, "allow_paid": False}, "output_dir": "runs"}

    def _scores(self, cfg):
        source = self.root / (cfg["name"] + ".yaml")
        source.write_text(yaml.safe_dump(cfg))
        directory = Path(execute(load_experiment(source))["run_dir"])
        return [{k: s[k] for k in ("source", "target", "positive_probability", "answers")}
                for row in read_jsonl(directory / "predictions.jsonl")
                for s in row["metadata"]["decision_scores"]], read_json(directory / "metrics.json")

    def test_parallel_workers_preserve_candidate_order_and_decisions(self):
        serial, serial_metrics = self._scores(self._mock_config("serial", max_workers=1))
        parallel, parallel_metrics = self._scores(self._mock_config("parallel", max_workers=8))
        self.assertEqual(serial, parallel)
        self.assertEqual(serial_metrics["summary"], parallel_metrics["summary"])

    def test_extra_questions_are_sent_together_and_recorded_without_changing_the_decision_key(self):
        extra = {"same_event": {"instructions": "Do both mentions refer to the same real-world event?",
                                "criteria": {"true": "Same event.", "false": "Different events."}}}
        scores, _ = self._scores(self._mock_config("multi", extra_questions=extra, max_workers=4))
        self.assertTrue(scores)
        for score in scores:
            self.assertEqual(set(score["answers"]), {"causal", "same_event"})
            self.assertEqual(score["answers"]["causal"], score["positive_probability"])

    def test_custom_noul_criteria_reach_the_request(self):
        from pipelines.jev.state_builder import questions_from_params
        q = questions_from_params({"instructions": "ESC question", "criteria": {"true": "T", "false": "F"}})
        self.assertEqual(q["causal"]["criteria"], {"true": "T", "false": "F"})
        self.assertEqual(q["causal"]["instructions"], "ESC question")
        with self.assertRaises(ValueError):
            questions_from_params({"extra_questions": {"causal": {"instructions": "x", "criteria": {"true": "a", "false": "b"}}}})

    def test_unordered_requests_preserve_every_ordered_candidate(self):
        cfg = self._mock_config("unordered", unordered=True, max_workers=4)
        source = self.root / "unordered.yaml"
        source.write_text(yaml.safe_dump(cfg))
        directory = Path(execute(load_experiment(source))["run_dir"])
        for row, original in zip(read_jsonl(directory / "predictions.jsonl"), self.suite["documents"]):
            scores = row["metadata"]["decision_scores"]
            expected = {tuple(pair) for pair in original["pair_list"]}
            self.assertEqual({(s["source"], s["target"]) for s in scores}, expected)
            self.assertEqual(row["metadata"]["query_count"], len({frozenset(pair) for pair in expected}))
            lookup = {(s["source"], s["target"]): s for s in scores}
            for a, b in expected:
                if (b, a) in lookup:
                    self.assertEqual(lookup[a, b]["positive_probability"], lookup[b, a]["positive_probability"])

    def test_cli_runs_a_campaign_of_configs(self):
        from pipelines.jev.cli import main
        first, second = self._mock_config("camp-a"), self._mock_config("camp-b", max_workers=4)
        for cfg in (first, second):
            (self.root / (cfg["name"] + ".yaml")).write_text(yaml.safe_dump(cfg))
        campaign = self.root / "campaign.yaml"
        campaign.write_text(yaml.safe_dump({"schema": "workere-campaign/v1", "experiments": [
            {"config": "camp-a.yaml"}, {"config": "camp-b.yaml"}]}))
        import os
        cwd = os.getcwd()
        os.chdir(self.root)
        try:
            self.assertEqual(main(["dry-run", "--config", "campaign.yaml", "--output", "plan.json"]), 0)
            self.assertEqual(main(["run", "--config", "campaign.yaml", "--output", "receipt.json"]), 0)
        finally:
            os.chdir(cwd)
        receipt = read_json(self.root / "receipt.json")
        self.assertEqual([r["status"] for r in receipt["runs"]], ["completed", "completed"])
        self.assertEqual(len(read_json(self.root / "plan.json")["plans"]), 2)


if __name__ == "__main__":
    unittest.main()
