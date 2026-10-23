import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipelines.jev.batch import BATCH_ENDPOINT, BatchChatClient
from pipelines.jev.client import BudgetExceeded


class BatchTransport(unittest.TestCase):
    def client(self, path, **overrides):
        return BatchChatClient(model="openai/gpt-6-luna:batch", model_version="gpt-6-luna-20260922",
            endpoint=BATCH_ENDPOINT, mode="live", cache_path=str(path), max_context_tokens=1050000,
            max_calls=100, budget_usd=.25, max_cost_per_call_usd=.015, input_price_per_million=.05,
            **overrides)

    def test_batch_response_mapping_cost_and_durable_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cache.sqlite"
            messages = [[{"role": "user", "content": str(i)}] for i in range(2)]
            budget = {"path": str(Path(temp) / "global.sqlite"), "scope": "test", "limit_usd": 1}
            events, calls, submitted = [], [], []
            def http(method, url, payload=None):
                calls.append(method)
                if method == "POST":
                    self.assertEqual(url, BATCH_ENDPOINT)
                    self.assertEqual(list(payload)[-1], "requests")
                    self.assertEqual(payload["model"], "openai/gpt-6-luna")
                    submitted.extend(payload["requests"])
                    return {"id": "batch_test", "status": "validating"}
                return {"id": "batch_test", "status": "completed", "usage": {"cost": .002},
                    "results": [{"custom_id": r["custom_id"], "error": None, "response": {"status_code": 200,
                        "body": {"model": "openai/gpt-6-luna-20260922", "choices": [{"finish_reason": "stop",
                                  "message": {"content": "[]"}}]}}} for r in reversed(submitted)]}
            with self.client(path, shared_budget=budget) as client, patch.object(client, "_http", http), patch("pipelines.jev.batch.time.sleep"):
                packets = client.decide_many(messages, [events.append] * 2)
                self.assertEqual(len(packets), 2)
                self.assertEqual(sum(e["cost_usd"] for e in events), .002)
                self.assertAlmostEqual(client.shared_budget.status()["accounted_or_reserved_usd"], .002)
                self.assertTrue(all(r["body"]["reasoning"]["effort"] == "high" for r in submitted))
                # Simulate a process exiting before caching the received responses.
                client.cache.execute("DELETE FROM decisions_v1"); client.cache.commit()
            with self.client(path, shared_budget=budget) as client, patch.object(client, "_http", side_effect=AssertionError("must recover receipt")):
                packets = client.decide_many(messages, [events.append] * 2)
                self.assertEqual(len(packets), 2)
                self.assertAlmostEqual(client.shared_budget.status()["accounted_or_reserved_usd"], .002)
            self.assertEqual(calls.count("POST"), 1)
            self.assertEqual(calls.count("GET"), 1)
            # A later plan with its own scope resumes the same receipt without charging it again.
            other = {**budget, "scope": "later-plan"}
            with self.client(path, shared_budget=other) as client, patch.object(client, "_http", side_effect=AssertionError("must recover receipt")):
                client.cache.execute("DELETE FROM decisions_v1"); client.cache.commit()
                resumed = []
                packets = client.decide_many(messages, [resumed.append] * 2)
                self.assertEqual(len(packets), 2)
                self.assertEqual(sum(e["cost_usd"] for e in resumed), 0)
                self.assertEqual(client.shared_budget.status()["accounted_or_reserved_usd"], 0)

    def test_uncertain_submission_cannot_be_submitted_twice(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cache.sqlite"
            messages = [[{"role": "user", "content": "one"}]]
            with self.client(path) as client, patch.object(client, "_http", side_effect=TimeoutError):
                with self.assertRaisesRegex(BudgetExceeded, "uncertain"):
                    client.decide_many(messages, [lambda e: None])
            with self.client(path) as client, patch.object(client, "_http", side_effect=AssertionError("no duplicate POST")):
                with self.assertRaisesRegex(BudgetExceeded, "uncertain"):
                    client.decide_many(messages, [lambda e: None])

    def test_interrupted_pending_batch_resumes_without_a_second_post_or_reservation(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cache.sqlite"
            messages = [[{"role": "user", "content": "pending"}]]
            budget = {"path": str(Path(temp) / "global.sqlite"), "scope": "same-plan", "limit_usd": 1}
            submitted = []
            def post(method, url, payload=None):
                self.assertEqual(method, "POST")
                submitted.extend(payload["requests"])
                return {"id": "batch_pending", "status": "in_progress"}
            with self.client(path, shared_budget=budget) as client, patch.object(client, "_http", post), \
                    patch.object(client, "_poll", side_effect=RuntimeError("controller stopped")):
                with self.assertRaisesRegex(RuntimeError, "controller stopped"):
                    client.decide_many(messages, [lambda e: None])
                self.assertAlmostEqual(client.shared_budget.status()["accounted_or_reserved_usd"], .015)
            def get(method, url, payload=None):
                self.assertEqual(method, "GET")
                self.assertTrue(url.endswith("batch_pending"))
                return {"id": "batch_pending", "status": "completed", "usage": {"cost": .002},
                    "results": [{"custom_id": r["custom_id"], "response": {"status_code": 200,
                        "body": {"model": "openai/gpt-6-luna-20260922", "choices": [{"finish_reason": "stop",
                                  "message": {"content": "[]"}}]}}} for r in submitted]}
            with self.client(path, shared_budget=budget) as client, patch.object(client, "_http", get), \
                    patch("pipelines.jev.batch.time.sleep"):
                events = []
                self.assertEqual(len(client.decide_many(messages, [events.append])), 1)
                self.assertFalse(events[0]["api_call"])
                self.assertAlmostEqual(client.shared_budget.status()["accounted_or_reserved_usd"], .002)


if __name__ == "__main__":
    unittest.main()
