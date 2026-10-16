import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark.interfaces import Context
from benchmark.schema import Document, Entity
from pipelines.jev.llm import ChatClient, LunaDocumentPredictor, marked_text


class LunaReference(unittest.TestCase):
    def test_seven_pass_vote_keeps_all_candidates_and_logs_effort(self):
        with tempfile.TemporaryDirectory() as temp:
            client = {"model": "openai/gpt-6-luna:batch", "model_version": "gpt-6-luna-20260922",
                      "endpoint": "https://openrouter.ai/api/v1/chat/completions", "mode": "mock",
                      "cache_path": str(Path(temp) / "cache.sqlite"), "max_calls": 10,
                      "reasoning_effort": "high", "max_context_tokens": 1050000}
            p = LunaDocumentPredictor(client, "prompts/esl_standard_eci_lean_coref.yaml",
                "experiments/jev/document-selection.json", "unused", "pinned")
            doc = Document("37_example", ("Fire", "damaged", "house"), ((0, 3),))
            entities = (Entity("1", (0,)), Entity("2", (1,)))
            context = Context(42, "relations_gold", doc.id)
            result = p.predict(doc, entities, (("1", "2"), ("2", "1")), context)
            self.assertEqual(len(result), 2)
            self.assertEqual([r.label for r in result], ["norel", "norel"])
            self.assertEqual(len(context.usage), 7)
            self.assertTrue(all(e["request"]["reasoning"] == {"effort": "high"} for e in context.usage))
            self.assertTrue(all(e["request"]["model"] == "openai/gpt-6-luna:batch" for e in context.usage))
            self.assertEqual(marked_text(doc, entities), "<1 Fire> <2 damaged> house")
            p.close()

    def test_truncation_and_wrong_model_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with ChatClient(model="openai/gpt-6-luna:batch", model_version="gpt-6-luna-20260922",
                            cache_path=str(Path(temp) / "cache.sqlite")) as client:
                for response in [
                    {"model": "mock/gpt-6-luna-20260922", "choices": [{"finish_reason": "length", "message": {"content": "[]"}}]},
                    {"model": "mock/gpt-6-luna-20260922", "choices": [{"finish_reason": "stop", "message": {"content": "incomplete"}}]}]:
                    with self.assertRaises(ValueError if response["choices"][0]["finish_reason"] == "stop" else RuntimeError):
                        client._validate_response({}, response)
                client.mode = "live"
                with self.assertRaises(RuntimeError):
                    client._validate_response({}, {"model": "other/model", "choices": []})


if __name__ == "__main__":
    unittest.main()
