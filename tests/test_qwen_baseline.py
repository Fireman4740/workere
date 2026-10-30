"""Local transport and pinned-data checks; no model or dataset downloads."""
import asyncio
import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx
import yaml
from langchain_core.messages import AIMessage, HumanMessage
from eci.llm import CallMeta

from benchmark.data import load_split

SCRIPT = Path(__file__).resolve().parents[1] / "experiments/jev/qwen/run.py"
loader = importlib.util.spec_from_file_location("qwen_baseline", SCRIPT)
qwen = importlib.util.module_from_spec(loader)
loader.loader.exec_module(qwen)


class QwenBaseline(unittest.TestCase):
    def test_resume_reuses_only_valid_pass_with_exact_prompt(self):
        messages = [HumanMessage(content="original prompt")]
        answer = '[{"pair":"a,b","label":"causes"}]'
        row = {"role": "inference", "doc_id": "doc", "pass": 2, "block": None,
               "ok": True, "finish_reason": "stop", "messages":
                   [m.to_json() for m in messages + [AIMessage(content=answer)]]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            truncated = {**row, "pass": 3, "ok": False, "finish_reason": "length"}
            path.write_text(json.dumps(row) + "\n" + json.dumps(truncated) + "\n")
            runner = qwen.SharedRunner.__new__(qwen.SharedRunner)
            runner.resume_passes = qwen.load_resume_passes(path)
            self.assertEqual(len(runner.resume_passes), 1)
            runner.prompt = SimpleNamespace(steps=None)
            runner.trace, runner.resumed_calls, runner.resume_source = Mock(), 0, "source"
            meta = CallMeta(role="inference", doc_id="doc", pass_idx=2)
            with patch.object(qwen.Runner, "_pass", new_callable=AsyncMock) as fresh:
                value = asyncio.run(runner._pass(messages, meta, None))
                self.assertEqual(value, (answer, [("a", "causes", "b")], []))
                fresh.assert_not_awaited()
                self.assertEqual(runner.resumed_calls, 1)
                self.assertEqual(runner.trace.write.call_args.args[0]["resumed_from_run"], "source")
                asyncio.run(runner._pass([HumanMessage(content="changed prompt")], meta, None))
                fresh.assert_awaited_once()

    def test_resume_requires_identical_scientific_configuration(self):
        spec = yaml.safe_load(qwen.SPEC.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            spec["state_root"] = tmp
            spec["only"] = ["r7_ctb"]
            old_state = Path(tmp) / spec["resume_evaluation_from_run"]
            old_dir = old_state / "r7_ctb" / "runs" / "prior"
            old_dir.mkdir(parents=True)
            (old_state / "r7_ctb/run-dir.txt").write_text(str(old_dir))
            old_cfg = qwen.effective_configs(spec, old_state)["r7_ctb"]
            (old_dir / "prior.config.yaml").write_text(yaml.safe_dump(old_cfg))
            # A stopped corpus can have no traces yet (all prior demo calls cached).
            (old_dir / "docs.jsonl").write_text(json.dumps({"id": "done", "failure": None}) + "\n"
                                                  + json.dumps({"id": "failed", "failure": "error"}) + "\n")
            state = Path(tmp) / "next-run"
            (state / "r7_ctb").mkdir(parents=True)
            (state / "resume-caches.json").write_text("{}")
            configs = qwen.effective_configs(spec, state)
            configs["r7_ctb"]["seed"] += 1
            with self.assertRaisesRegex(ValueError, "configuration change"):
                qwen.restore_evaluation(spec, configs, state)
            configs = qwen.effective_configs(spec, state)
            qwen.restore_evaluation(spec, configs, state)
            saved = (state / "r7_ctb/resume-docs.jsonl").read_text().splitlines()
            self.assertEqual([json.loads(row)["id"] for row in saved], ["done"])
            self.assertEqual((state / "r7_ctb/resume-traces.jsonl").read_text(), "")

    def test_waits_for_model_service_before_inference(self):
        spec = yaml.safe_load(qwen.SPEC.read_text())
        calls = []

        def respond(request):
            calls.append(request)
            self.assertEqual(request.method, "GET")
            self.assertTrue(request.url.path.endswith("/models"))
            self.assertEqual(request.headers["authorization"], "Bearer test-key")
            if len(calls) == 1:
                return httpx.Response(503)
            return httpx.Response(200, json={"data": [{"id": spec["model"]}]})

        async def run():
            async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
                with patch.object(qwen.asyncio, "sleep", new_callable=AsyncMock) as sleep:
                    await qwen.wait_for_api(spec, "test-key", http)
                    sleep.assert_awaited_once()

        asyncio.run(run())
        self.assertEqual(len(calls), 2)

    def test_demo_cache_reuse_requires_identical_sampling(self):
        spec = yaml.safe_load(qwen.SPEC.read_text())
        spec["resume_cache_from_run"] = "previous-run"
        with tempfile.TemporaryDirectory() as tmp:
            spec["state_root"] = tmp
            source = Path(tmp) / spec["resume_cache_from_run"]
            source.mkdir()
            (source / "transport.yaml").write_text(yaml.safe_dump(spec))
            corpus = source / "r7_ctb"
            corpus.mkdir()
            with sqlite3.connect(corpus / "llm-cache.sqlite") as db:
                db.execute("CREATE TABLE llm_cache (key TEXT PRIMARY KEY, generations TEXT)")
                db.execute("INSERT INTO llm_cache VALUES ('same-prompt', 'saved-result')")
            destination = Path(tmp) / "new-run"
            qwen.restore_demo_caches(spec, destination)
            with sqlite3.connect(destination / "r7_ctb/llm-cache.sqlite") as db:
                self.assertEqual(db.execute("SELECT generations FROM llm_cache").fetchone(), ("saved-result",))
            receipt = json.loads((destination / "resume-caches.json").read_text())
            self.assertEqual(receipt["corpora"][0]["cached_calls_restored"], 1)
            self.assertFalse(receipt["evaluation_predictions_reused"])
            with self.assertRaisesRegex(ValueError, "sampling"):
                qwen.restore_demo_caches({**spec, "temperature": 0.5}, Path(tmp) / "different-run")
            changed = {**spec, "base_url": "http://new-local-route:18020/v1", "resume_allow_endpoint_change": False}
            with self.assertRaisesRegex(ValueError, "Endpoint change"):
                qwen.restore_demo_caches(changed, Path(tmp) / "blocked-endpoint")
            qwen.restore_demo_caches({**changed, "resume_allow_endpoint_change": True}, Path(tmp) / "allowed-endpoint")
            with self.assertRaisesRegex(ValueError, "sampling"):
                qwen.restore_demo_caches({**changed, "resume_allow_endpoint_change": True, "temperature": 0.5},
                                         Path(tmp) / "blocked-sampling")

    def test_transport_uses_requested_temperature_zero_and_keeps_reasoning(self):
        spec = yaml.safe_load(qwen.SPEC.read_text())
        captured = []

        def respond(request):
            body = json.loads(request.content)
            captured.append(body)
            self.assertEqual(request.headers["authorization"], "Bearer test-key")
            return httpx.Response(200, json={
                "id": "test", "object": "chat.completion", "created": 1,
                "model": spec["model"], "choices": [{"index": 0, "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "[]",
                                "reasoning_content": "Reasoning retained for audit."}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}})

        async def run():
            async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
                model = qwen.model_factory(spec, "test-key", client)(spec["model"], 0.0, "xhigh", False)
                answer = await model.ainvoke([HumanMessage(content="test")])
                self.assertEqual(answer.content, "[]")
                self.assertEqual(answer.additional_kwargs["reasoning_content"], "Reasoning retained for audit.")

        asyncio.run(run())
        self.assertEqual(len(captured), 1)
        body = captured[0]
        self.assertEqual(body["model"], "qwen3.8-27b")
        self.assertEqual(body["temperature"], 0.0)
        self.assertEqual(body["reasoning_effort"], "xhigh")
        self.assertEqual(body["top_p"], .95)
        self.assertEqual(body["top_k"], 20)
        self.assertEqual(body["min_p"], 0.0)
        self.assertEqual(body["presence_penalty"], 0.0)
        self.assertEqual(body["repetition_penalty"], 1.0)
        self.assertEqual(body["max_tokens"], 32768)
        self.assertEqual(body["stream"], False)
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": True, "preserve_thinking": True})
        for key in ("provider", "reasoning", "max_completion_tokens", "tools"):
            self.assertNotIn(key, body)

    def test_revision_is_forwarded_to_dataset_loader(self):
        with patch("datasets.load_dataset", return_value=[]) as load:
            self.assertEqual(load_split("test/repo", "train", text_field="text", ann_field="annots",
                valid_labels=None, binary_undirected=False, shuffle_pair_list=False, revision="frozen"), [])
        load.assert_called_once_with("test/repo", split="train", revision="frozen")

    def test_only_requested_corpora_use_original_best_protocol(self):
        spec = yaml.safe_load(qwen.SPEC.read_text())
        configs = qwen.effective_configs(spec, Path("/tmp/qwen-test"))
        self.assertEqual(set(configs), {"r7_ctb", "r7_esc"})
        for cfg in configs.values():
            self.assertEqual(cfg["model"]["reasoning_effort"], "xhigh")
            self.assertEqual(cfg["model"]["temperature"], 0.0)
            self.assertTrue(cfg["few_shot"]["cot_generation"]["enabled"])
            self.assertTrue(cfg["few_shot"]["cot_generation"]["rewrite"])
            self.assertEqual(cfg["few_shot"]["n_examples"], 1)
            self.assertEqual(cfg["experiment"]["resampling"]["n_runs"], 7)
            self.assertFalse(cfg["experiment"]["enable_tools"])
            self.assertFalse(cfg["coref_hint"]["enabled"])
            self.assertFalse(cfg["gold_count"]["enabled"])


if __name__ == "__main__":
    unittest.main()
