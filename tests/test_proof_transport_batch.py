"""Mock Batch API tests. Never submit to a provider or approve real spending."""

from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from proof_transport import client
from proof_transport.batch import ENDPOINT, Store
from proof_transport.fixtures import raw_fixture
from proof_transport.pipeline import (
    extraction_requests,
    load_plan,
    prepare,
    read_jsonl,
    request,
)
from proof_transport.schema import Invalid, digest


class FakeAPI:
    def __init__(self):
        self.jobs = {}
        self.calls = []
        self.pending = False
        self.fail_post = False
        self.fail_get = False
        self.bad_ids = False
        self.truncated = False
        self.no_usage = False
        self.aggregate_only = False
        self.aggregate_cost = 0.000001
        self.tokens = (11, 7)
        self.versioned = True

    def __call__(self, method, url, payload=None):
        self.calls.append((method, url, deepcopy(payload)))
        if method == "POST":
            if self.fail_post:
                raise TimeoutError("mock unknown submission")
            bid = "batch_" + str(len(self.jobs) + 1)
            self.jobs[bid] = deepcopy(payload)
            return {"id": bid, "status": "validating"}
        if self.fail_get:
            raise TimeoutError("mock retrieval timeout")
        bid = url.rsplit("/", 1)[-1]
        payload = self.jobs[bid]
        if self.pending:
            return {"id": bid, "status": "in_progress"}
        rows = []
        for item in payload["requests"]:
            body = {
                "model": "openai/gpt-6-luna-20260922"
                if self.versioned
                else "openai/gpt-6-luna",
                "choices": [
                    {
                        "finish_reason": "length" if self.truncated else "stop",
                        "message": {"content": json.dumps({"items": []})},
                    }
                ],
            }
            if not self.no_usage and not self.aggregate_only:
                body["usage"] = {
                    "prompt_tokens": self.tokens[0],
                    "completion_tokens": self.tokens[1],
                }
            rows.append(
                {
                    "custom_id": item["custom_id"],
                    "error": None,
                    "response": {"status_code": 200, "body": body},
                }
            )
        if self.bad_ids:
            rows[0]["custom_id"] = "unexpected"
        usage = {"cost": self.aggregate_cost * len(rows)}
        if self.aggregate_only and not self.no_usage:
            usage.update(
                prompt_tokens=len(rows) * self.tokens[0],
                completion_tokens=len(rows) * self.tokens[1],
            )
        return {
            "id": bid,
            "status": "completed",
            "usage": usage,
            "results": list(reversed(rows)),
        }


class BatchPilot(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        suite, controls = raw_fixture()
        prepare(suite, controls, self.path / "cache", self.path / "plan")
        self.manifest, self.bundle, self.roots, _, self.config = load_plan(
            self.path / "plan"
        )
        self.requests = extraction_requests(self.path / "plan")
        catalog = {
            "data": [
                {
                    "id": "openai/gpt-6-luna:batch",
                    "canonical_slug": "openai/gpt-6-luna-20260922",
                    "pricing": {
                        "prompt": "0.00000005",
                        "completion": "0.00000025",
                        "overrides": [
                            {
                                "min_prompt_tokens": 272000,
                                "prompt": "0.0000001",
                                "completion": "0.000000375",
                            }
                        ],
                    },
                    "context_length": 1050000,
                    "supported_parameters": [
                        "max_tokens",
                        "reasoning",
                        "response_format",
                    ],
                }
            ]
        }
        self.pricing = client.tariff_snapshot(json.dumps(catalog), self.config["model"])
        self.ledger = self.path / "budget.sqlite"
        self.output = self.path / "responses.jsonl"
        self.approval = {
            "approval": "APPROVED_BY_USER",
            "approved_by": "UNIT_TEST_NOT_AN_OPERATOR_APPROVAL",
            "plan_hash": digest(self.manifest),
            "config_hash": digest(self.config),
            "pricing_hash": digest(self.pricing),
            "model": self.config["model"],
            "max_usd": 10.0,
            "max_billed_tokens": 1500000,
            "expires_utc": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
            "ledger_path": str(self.ledger.resolve()),
        }
        self.api = FakeAPI()

    def submit(self, **kwargs):
        return client.execute(
            self.requests,
            self.manifest,
            self.config,
            self.pricing,
            self.approval,
            self.ledger,
            self.output,
            http=kwargs.pop("http", self.api),
            **kwargs,
        )

    def collect(self, **kwargs):
        return client.collect(
            self.requests,
            self.manifest,
            self.config,
            self.pricing,
            self.approval,
            self.ledger,
            self.output,
            http=kwargs.pop("http", self.api),
            **kwargs,
        )

    def store(self):
        s = Store(self.ledger, digest(self.manifest), self.approval)
        self.addCleanup(s.close)
        return s

    def add_requests(self, n, stage="extract"):
        doc = next(iter(self.bundle["documents"].values()))
        self.requests = [
            request(
                stage,
                doc["id"],
                doc,
                [{"source_id": f"unit_{i}", "pair": ["a", "b"]}],
                self.manifest,
                self.config,
            )
            for i in range(n)
        ]

    def test_explicit_batch_tariff_not_inferred_discount(self):
        self.assertEqual(self.pricing["model"], "openai/gpt-6-luna:batch")
        self.assertEqual(self.pricing["transport"], "batch_catalog")
        self.assertEqual(self.pricing["conservative_prompt_usd_per_token"], "1E-7")
        self.assertEqual(
            self.pricing["conservative_completion_usd_per_token"], "3.75E-7"
        )
        with self.assertRaises(Invalid):
            client.tariff_snapshot(json.dumps({"data": []}), self.config["model"])

    def test_true_batch_routing_and_no_sync_fallback(self):
        result = self.submit()
        self.assertEqual(result["new_batch_posts"], 1)
        self.assertFalse(self.output.exists())
        method, url, payload = self.api.calls[0]
        self.assertEqual((method, url), ("POST", ENDPOINT))
        self.assertEqual(list(payload)[-1], "requests")
        self.assertEqual(payload["model"], "openai/gpt-6-luna")
        self.assertEqual(payload["endpoint"], "/v1/chat/completions")
        self.assertEqual(payload["completion_window"], "24h")
        body = payload["requests"][0]["body"]
        self.assertNotIn("model", body)
        self.assertEqual(body["reasoning"]["effort"], "high")
        self.assertFalse(body["provider"]["allow_fallbacks"])

    def test_complete_reordered_results_joined_by_custom_id(self):
        self.add_requests(3)
        self.submit()
        self.collect()
        receipts = read_jsonl(self.output)
        self.assertEqual(
            [r["request_id"] for r in receipts], [r["id"] for r in self.requests]
        )
        self.assertEqual(sum(r["usage"]["input_tokens"] for r in receipts), 33)
        self.assertAlmostEqual(sum(r["usage"]["cost_usd"] for r in receipts), 0.000003)
        self.assertTrue(all(r["version_verified"] for r in receipts))
        self.assertTrue(all(r["model"].endswith(":batch") for r in receipts))

    def test_resume_acknowledged_job_never_posts_again(self):
        self.submit()
        result = self.submit(http=lambda *a: self.fail("must not POST again"))
        self.assertEqual(result["new_batch_posts"], 0)
        self.collect()
        self.assertEqual([x[0] for x in self.api.calls], ["POST", "GET"])

    def test_settled_receipts_recovered_without_any_http(self):
        self.submit()
        self.collect()
        original = self.output.read_text()
        self.output.unlink()
        self.collect(http=lambda *a: self.fail("must recover durable receipts"))
        self.assertEqual(self.output.read_text(), original)

    def test_ambiguous_submission_keeps_reservation_and_never_resubmits(self):
        self.api.fail_post = True
        with self.assertRaises(TimeoutError):
            self.submit()
        with self.assertRaises(Invalid):
            self.submit(http=lambda *a: self.fail("no duplicate POST"))
        s = self.store()
        row = s.db.execute("SELECT status,tokens FROM request").fetchone()
        self.assertEqual(row[0], "RESERVED")
        self.assertGreater(row[1], 0)

    def test_missing_batch_id_never_resubmits(self):
        with self.assertRaises(Invalid):
            self.submit(http=lambda *a: {"status": "validating"})
        with self.assertRaises(Invalid):
            self.submit(http=lambda *a: self.fail("no second POST"))

    def test_collection_pending_writes_no_fake_final_output(self):
        self.submit()
        self.api.pending = True
        r = self.collect()
        self.assertEqual(r["status"], "PENDING")
        self.assertFalse(self.output.exists())
        self.api.pending = False
        self.collect()
        self.assertEqual(sum(c[0] == "POST" for c in self.api.calls), 1)

    def test_get_timeout_is_resumable_by_id_without_resubmission(self):
        self.submit()
        self.api.fail_get = True
        with self.assertRaises(TimeoutError):
            self.collect()
        self.api.fail_get = False
        self.collect()
        self.assertEqual(sum(c[0] == "POST" for c in self.api.calls), 1)

    def test_cached_batch_receipts_need_no_api_key_or_network(self):
        self.submit()
        self.collect()
        self.output.unlink()
        with (
            patch.dict("os.environ", {}, clear=True),
            patch(
                "urllib.request.urlopen", side_effect=AssertionError("no GET needed")
            ),
        ):
            self.collect(http=None)
        self.assertEqual(len(read_jsonl(self.output)), 1)

    def test_malformed_ids_still_record_known_higher_batch_debit(self):
        self.submit()
        self.api.bad_ids = True
        self.api.aggregate_cost = 20.0
        with self.assertRaises(Invalid):
            self.collect()
        self.assertEqual(
            self.store().db.execute("SELECT usd FROM request").fetchone()[0], 20.0
        )
        self.assertFalse(self.output.exists())

    def test_expired_approval_can_collect_but_cannot_submit_new_job(self):
        self.submit()
        self.approval["expires_utc"] = (
            datetime.now(UTC) - timedelta(hours=1)
        ).isoformat()
        self.collect()
        self.add_requests(2)
        with self.assertRaises(Invalid):
            self.submit(http=lambda *a: self.fail("expired approval cannot POST"))

    def test_old_synchronous_price_or_endpoint_refused(self):
        self.pricing["transport"] = "synchronous_no_batch_discount"
        self.approval["pricing_hash"] = digest(self.pricing)
        with self.assertRaises(Invalid):
            self.submit(http=lambda *a: self.fail("wrong transport must not POST"))

    def test_source_stage_and_arm_stages_are_distinct_jobs(self):
        doc = next(iter(self.bundle["documents"].values()))
        self.requests = [
            request(
                v, doc["id"], doc, [{"candidate_id": "x"}], self.manifest, self.config
            )
            for v in ("A0", "A1", "A2")
        ]
        result = self.submit()
        self.assertEqual(result["new_batch_posts"], 3)
        self.assertEqual(
            len(
                {
                    j[2]["requests"][0]["body"]["messages"][0]["content"]
                    for j in self.api.calls
                }
            ),
            3,
        )

    def test_rechunking_subset_cannot_submit_request_again(self):
        self.add_requests(10)
        self.submit()
        posts = len(self.api.calls)
        self.requests = self.requests[1:4]
        self.submit(
            http=lambda *a: self.fail("assigned request cannot move to another batch")
        )
        self.assertEqual(len(self.api.calls), posts)

    def test_reserve_whole_batch_atomically_before_post(self):
        self.add_requests(2)
        self.approval["max_billed_tokens"] = 1
        result = self.submit(http=lambda *a: self.fail("cannot POST over token budget"))
        self.assertEqual(result["assigned_requests"], 0)
        self.assertEqual(
            self.store().db.execute("SELECT COUNT(*) FROM request").fetchone()[0], 0
        )

    def test_partial_submission_does_not_forge_complete_coverage(self):
        self.add_requests(9)
        bound = client.upper_bound(self.requests[0], self.pricing)[
            "billed_token_upper_bound"
        ]
        self.approval["max_billed_tokens"] = bound * 8
        result = self.submit()
        self.assertEqual(result["assigned_requests"], 8)
        r = self.collect()
        self.assertEqual(r["status"], "PENDING")
        self.assertEqual(len(r["unassigned_requests"]), 1)
        self.assertFalse(self.output.exists())
        # Settled real usage frees headroom for the remaining immutable request.
        self.submit()
        self.collect()
        self.assertEqual(len(read_jsonl(self.output)), 9)

    def test_missing_tokens_cannot_be_filled_with_zero(self):
        self.submit()
        self.api.no_usage = True
        with self.assertRaisesRegex(Invalid, "MISSING_BATCH_USAGE"):
            self.collect()
        self.assertFalse(self.output.exists())
        self.assertEqual(
            self.store().db.execute("SELECT status FROM request").fetchone()[0],
            "RESERVED",
        )
        with self.assertRaises(Invalid):
            self.submit(http=lambda *a: self.fail("blocked job cannot be resubmitted"))

    def test_aggregate_tokens_and_cost_allocation_is_explicit(self):
        self.add_requests(3)
        self.submit()
        self.api.aggregate_only = True
        self.collect()
        rows = read_jsonl(self.output)
        self.assertEqual(sum(r["usage"]["input_tokens"] for r in rows), 33)
        self.assertEqual(sum(r["usage"]["output_tokens"] for r in rows), 21)
        self.assertTrue(
            all("not_per_request_measurement" in r["token_usage_kind"] for r in rows)
        )
        self.assertAlmostEqual(sum(r["usage"]["cost_usd"] for r in rows), 0.000003)

    def test_bad_result_ids_block_and_keep_budget(self):
        self.submit()
        self.api.bad_ids = True
        with self.assertRaisesRegex(Invalid, "BATCH_RESULT_IDS"):
            self.collect()
        self.assertFalse(self.output.exists())
        self.assertEqual(
            self.store().db.execute("SELECT status FROM proof_batch_job").fetchone()[0],
            "BLOCKED",
        )

    def test_truncation_accounts_spend_but_never_scores(self):
        self.submit()
        self.api.truncated = True
        with self.assertRaisesRegex(Invalid, "FAILED_BATCH_RESULTS_CHARGED"):
            self.collect()
        s = self.store()
        usd, tokens, status = s.db.execute(
            "SELECT usd,tokens,status FROM request"
        ).fetchone()
        self.assertAlmostEqual(usd, 0.000001)
        self.assertEqual(tokens, 18)
        self.assertEqual(status, "DONE")
        self.assertFalse(self.output.exists())
        with self.assertRaises(Invalid):
            self.submit(http=lambda *a: self.fail("no automatic retry"))

    def test_actual_overspend_is_recorded_before_scope_stops(self):
        self.submit()
        self.api.aggregate_cost = 20.0
        with self.assertRaisesRegex(Invalid, "PROVIDER_EXCEEDED_BATCH_RESERVATION"):
            self.collect()
        s = self.store()
        self.assertEqual(
            s.db.execute("SELECT usd,status FROM request").fetchone(),
            (20.0, "VIOLATION"),
        )
        self.add_requests(2)
        with self.assertRaises(Invalid):
            self.submit(http=lambda *a: self.fail("cannot spend after violation"))

    def test_no_authorization_means_no_batch_post(self):
        self.approval["approval"] = "NOT_AUTHORIZED"
        with (
            patch("urllib.request.urlopen", side_effect=AssertionError("no network")),
            self.assertRaises(Invalid),
        ):
            self.submit(http=None)
        self.assertFalse(self.ledger.exists())


if __name__ == "__main__":
    unittest.main()
