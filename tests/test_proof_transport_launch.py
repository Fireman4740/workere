"""Lightweight controller checks; no network, no provider and no real ledger."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.jev.proof_transport_launch.controller import (
    Context,
    drive_stage,
    load_recipe,
)
from proof_transport.pipeline import write_jsonl_once
from proof_transport.schema import Invalid


class LaunchController(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.state = Path(self.temp.name)
        self.recipe = {"max_phase_wait_seconds": 10, "poll_interval_seconds": 0}
        self.context = Context(
            (
                self.recipe,
                None,
                {},
                None,
                None,
                None,
                {},
                {},
                {"ledger_path": str(self.state / "fake.sqlite")},
                self.state,
            )
        )

    def test_committed_inputs_match_frozen_plan_without_gold_or_network(self):
        with patch("urllib.request.urlopen", side_effect=AssertionError("no network")):
            (
                recipe,
                _plan,
                _manifest,
                bundle,
                roots,
                _groups,
                config,
                _pricing,
                approval,
                _state,
            ) = load_recipe()
        self.assertEqual(len(roots), 120)
        self.assertEqual(config["model"], "openai/gpt-6-luna:batch")
        self.assertEqual(approval["max_billed_tokens"], 1500000)
        self.assertNotIn("gold", bundle)
        self.assertEqual(recipe["expected_plan_hash"], approval["plan_hash"])

    def test_stage_collects_after_submission_and_never_uses_missing_as_negative(self):
        calls = []

        def collect(*args):
            calls.append("GET")
            if len(calls) == 1:
                return {
                    "final_receipts_written": False,
                    "unassigned_requests": ["r"],
                    "pending_batch_ids": [],
                }
            write_jsonl_once(args[6], [{"request_id": "r"}])
            return {
                "final_receipts_written": True,
                "unassigned_requests": [],
                "pending_batch_ids": [],
            }

        def submit(*args):
            calls.append("POST")
            return {"new_batch_posts": 1}

        result = drive_stage(
            "test",
            [{"id": "r"}],
            self.context,
            collect_fn=collect,
            submit_fn=submit,
            sleep_fn=lambda _: None,
        )
        self.assertEqual(calls, ["GET", "POST", "GET"])
        self.assertEqual(result, [{"request_id": "r"}])
        self.assertFalse((self.state / "fake.sqlite").exists())

    def test_no_headroom_stops_without_partial_scoring(self):
        def collect(*args):
            return {
                "final_receipts_written": False,
                "unassigned_requests": ["r"],
                "pending_batch_ids": [],
            }

        def submit(*args):
            return {"new_batch_posts": 0}

        with self.assertRaisesRegex(Invalid, "BUDGET_LIMIT_INCOMPLETE"):
            drive_stage(
                "test",
                [{"id": "r"}],
                self.context,
                collect_fn=collect,
                submit_fn=submit,
                sleep_fn=lambda _: None,
            )
        self.assertFalse((self.state / "test.receipts.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
