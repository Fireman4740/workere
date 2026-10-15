"""Noul end-to-end and API safety tests. Mock results never stand in for model measurements."""
import copy
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
from pipelines.jev.state_builder import build_state, noul_question
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


if __name__ == "__main__":
    unittest.main()
