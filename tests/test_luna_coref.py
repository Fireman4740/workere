import json
import tempfile
import unittest
from pathlib import Path

from benchmark.interfaces import Context
from benchmark.schema import Document, Entity
from pipelines.jev.coref import LunaCorefPredictor, parse_groups


class LunaCoref(unittest.TestCase):
    def test_parse_groups_validates_ids_and_disjointness(self):
        ids = {"1", "2", "3", "4"}
        text = 'Step 1 ... final:\n[["1", "3"], ["2"], ["4", "2"]]'
        with self.assertRaises(ValueError):
            parse_groups(text, ids)  # "2" appears twice
        self.assertEqual(parse_groups('[["1","3"],["2"]]', ids), [["1", "3"]])  # singletons dropped
        self.assertEqual(parse_groups("no groups: []", ids), [])
        with self.assertRaises(ValueError):
            parse_groups('[["1","9"]]', ids)  # unknown mention id
        with self.assertRaises(ValueError):
            parse_groups("no json at all", ids)

    def test_mock_predictor_records_groups_and_emits_only_norel(self):
        with tempfile.TemporaryDirectory() as temp:
            client = {"model": "openai/gpt-6-luna:batch", "model_version": "gpt-6-luna-20260922",
                      "endpoint": "https://openrouter.ai/api/v1/batches", "mode": "mock",
                      "cache_path": str(Path(temp) / "cache.sqlite"), "max_calls": 4,
                      "reasoning_effort": "high", "max_context_tokens": 1050000}
            p = LunaCorefPredictor(client)
            doc = Document("37_example", ("Fire", "damaged", "the", "blaze"), ((0, 4),))
            entities = (Entity("1", (0,)), Entity("2", (1,)), Entity("3", (3,)))
            context = Context(42, "relations_gold", doc.id)
            out = p.predict(doc, entities, (("1", "2"), ("2", "1"), ("1", "3")), context)
            self.assertEqual({r.label for r in out}, {"norel"})
            self.assertEqual(context.prediction_metadata["coref_groups"], [])
            self.assertEqual(len(context.usage), 1)
            p.close()


if __name__ == "__main__":
    unittest.main()
