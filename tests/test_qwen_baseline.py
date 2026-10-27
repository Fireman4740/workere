"""Local transport and pinned-data checks; no model or dataset downloads."""
import asyncio
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import httpx
import yaml
from langchain_core.messages import HumanMessage

from benchmark.data import load_split

SCRIPT = Path(__file__).resolve().parents[1] / "experiments/jev/qwen/run.py"
loader = importlib.util.spec_from_file_location("qwen_baseline", SCRIPT)
qwen = importlib.util.module_from_spec(loader)
loader.loader.exec_module(qwen)


class QwenBaseline(unittest.TestCase):
    def test_transport_uses_official_thinking_sampling_and_keeps_reasoning(self):
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
                model = qwen.model_factory(spec, "test-key", client)(spec["model"], 1.0, "xhigh", False)
                answer = await model.ainvoke([HumanMessage(content="test")])
                self.assertEqual(answer.content, "[]")
                self.assertEqual(answer.additional_kwargs["reasoning_content"], "Reasoning retained for audit.")

        asyncio.run(run())
        self.assertEqual(len(captured), 1)
        body = captured[0]
        self.assertEqual(body["model"], "qwen3.8-27b")
        self.assertEqual(body["temperature"], 1.0)
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
            self.assertEqual(cfg["model"]["temperature"], 1.0)
            self.assertTrue(cfg["few_shot"]["cot_generation"]["enabled"])
            self.assertTrue(cfg["few_shot"]["cot_generation"]["rewrite"])
            self.assertEqual(cfg["few_shot"]["n_examples"], 1)
            self.assertEqual(cfg["experiment"]["resampling"]["n_runs"], 7)
            self.assertFalse(cfg["experiment"]["enable_tools"])
            self.assertFalse(cfg["coref_hint"]["enabled"])
            self.assertFalse(cfg["gold_count"]["enabled"])


if __name__ == "__main__":
    unittest.main()
