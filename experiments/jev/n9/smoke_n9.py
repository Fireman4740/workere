"""Synthetic controller+independent-auditor integration, zero network/model calls.

Synthetic controls are all-negative, d1=0, never copied from target gold. The
output proves software integration only and has no empirical performance value.
"""
import contextlib
import importlib.util
import io
import json
import os
import pathlib
import shutil
import tempfile
import uuid
from unittest.mock import patch

from pipelines.jev import n9
from pipelines.jev.n9_transport import FreshLogicalBatchClient


def main():
    run = "synthetic-n9-" + uuid.uuid4().hex
    root = pathlib.Path("logs/n9") / run
    actual = n9.P
    post_count = []

    def fake_http(client, method, url, payload=None):
        assert method == "POST", "completed fake batches need no GET"
        post_count.append(len(payload["requests"]))
        return {"id": "synthetic-batch-" + payload["requests"][0]["custom_id"],
                "status": "completed", "usage": {"cost": 0},
                "results": [{"custom_id": row["custom_id"],
                    "response": {"status_code": 200,
                        "body": {"id": "synthetic-response-" + row["custom_id"],
                                 "model": "openai/gpt-6-luna", "provider": "OpenAI",
                                 "choices": [{"finish_reason": "stop", "message": {"content": "[]"}}],
                                 "usage": {"prompt_tokens": 0, "completion_tokens": 0}}}}
                    for row in payload["requests"]]}

    try:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = pathlib.Path(tmp)
            for name in ("protocol.json", "public.json", "selection.json", "preflight-frozen.json",
                         "requests-frozen.json", "test_n9.py", "audit_n9.py"):
                shutil.copyfile(actual / name, fixture / name)
            controls = [{"doc": d["id"], "a": a, "b": b, "d1": 0., "gold": False}
                        for d in n9.read(actual / "public.json") for a, b in d["pair_list"]]
            n9.save(fixture / "controls.json", controls)
            n9.save(fixture / "d1-public.json", [{k: row[k] for k in ("doc", "a", "b", "d1")}
                                                for row in controls])
            with patch.object(n9, "P", fixture), \
                 patch.object(n9, "catalog", lambda p: n9.save(p / "catalog-response.json", {"synthetic": True})), \
                 patch.object(FreshLogicalBatchClient, "_http", fake_http), \
                 patch.dict(os.environ, {"ORX_RUN_ID": run, "ORX_COMMIT": "synthetic-only"}), \
                 contextlib.redirect_stdout(io.StringIO()):
                n9.main()
            spec = importlib.util.spec_from_file_location("independent_n9", actual / "audit_n9.py")
            auditor = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(auditor)
            replay = auditor.run(root)
            assert replay["audit"] == "PASS" and replay["raw_answers"] == 240
            assert len(post_count) == 10 and sum(post_count) == 240
            assert replay["provider_reported_cost_usd"] == 0
            print(json.dumps({"synthetic_integration": "PASS", "API_calls": 0,
                              "mock_POSTs": len(post_count), "logical_answers": 240,
                              "independent_raw_replay": "PASS",
                              "performance_value": "none; all answers and controls synthetic"}, indent=2))
    finally:
        if root.exists():
            shutil.rmtree(root)


if __name__ == "__main__":
    main()
