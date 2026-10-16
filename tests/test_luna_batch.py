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
            with self.client(path) as client, patch.object(client, "_http", http), patch("pipelines.jev.batch.time.sleep"):
                packets = client.decide_many(messages, [events.append] * 2)
                self.assertEqual(len(packets), 2)
                self.assertEqual(sum(e["cost_usd"] for e in events), .002)
                self.assertTrue(all(r["body"]["reasoning"]["effort"] == "high" for r in submitted))
                # Simulate a process exiting before caching the received responses.
                client.cache.execute("DELETE FROM decisions_v1"); client.cache.commit()
            with self.client(path) as client, patch.object(client, "_http", side_effect=AssertionError("must recover receipt")):
                packets = client.decide_many(messages, [events.append] * 2)
                self.assertEqual(len(packets), 2)
            self.assertEqual(calls.count("POST"), 1)
            self.assertEqual(calls.count("GET"), 1)

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


if __name__ == "__main__":
    unittest.main()
