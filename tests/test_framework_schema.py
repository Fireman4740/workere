"""Public schema and plugin contract tests; no model or dataset dependencies."""
import dataclasses
import unittest

from benchmark.registry import Registry
from benchmark.schema import Document, Entity, PipelineInput, Prediction, Relation


class Schema(unittest.TestCase):
    def setUp(self):
        self.doc = Document("d", ["Heavy", "rain", "caused", "flooding", "."], [[0, 5]])

    def test_roundtrip_discontinuous_offsets_and_confidence(self):
        pred = Prediction("d", (Entity("e0", (0, 1, 3), confidence=.8),),
                          (Relation("e0", "ghost", "causes", .7),))
        row = pred.to_dict(self.doc)
        self.assertEqual(row["entities"][0]["char_offsets"], [[0, 10], [18, 26]])
        self.assertEqual(Prediction.from_dict(row, self.doc), pred)
        row["entities"][0]["char_offsets"] = [[0, 11]]
        with self.assertRaisesRegex(ValueError, "offsets"):
            Prediction.from_dict(row, self.doc)

    def test_invalid_spans_ids_confidences_and_coverage(self):
        for span in ((), (True,), (-1,), (2, 1), (1, 1)):
            with self.assertRaises(ValueError):
                Entity("e", span)
        for value in (float("nan"), float("inf"), 1.1, -.1, True):
            with self.assertRaises(ValueError):
                Relation("a", "b", "causes", value)
        with self.assertRaisesRegex(ValueError, "bounds"):
            Prediction("d", (Entity("e", (99,)),)).validate(self.doc)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            Prediction("d", (Entity("e", (0,)), Entity("e", (1,)))).validate(self.doc)
        with self.assertRaisesRegex(ValueError, "mismatch"):
            Prediction("other").validate(self.doc)

    def test_document_and_entity_inputs_are_immutable_and_no_gold(self):
        with self.assertRaises(dataclasses.FrozenInstanceError):
            self.doc.lang = "fra"
        self.assertIsInstance(self.doc.tokens, tuple)
        self.assertNotIn("gold", self.doc.to_dict())
        raw = PipelineInput(self.doc).to_dict()
        self.assertNotIn("oracle", raw)
        self.assertNotIn("entities", raw)
        oracle = PipelineInput(self.doc, (Entity("a", (1,)),), ()).to_dict()
        self.assertEqual(oracle["oracle"]["kind"], "reference_mentions")
        self.assertNotIn("relations", oracle["oracle"])


class Plugins(unittest.TestCase):
    def test_duplicate_version_and_unknown_plugins_rejected(self):
        registry = Registry()
        registry.register("entity_extractor", "toy", version="v1")(object)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            registry.register("entity_extractor", "toy")(object)
        for spec in ({"plugin": "missing", "version": "v1"}, {"plugin": "toy", "version": "bad"},
                     {"plugin": "toy"}):
            with self.assertRaises(ValueError):
                registry.build("entity_extractor", spec, {})

    def test_network_and_paid_guard_precedes_factory(self):
        registry, constructed = Registry(), []

        @registry.register("relation_predictor", "billable", requires_network=True, may_charge=True)
        def factory():
            constructed.append(True)
            return object()

        spec = {"plugin": "billable", "version": "1"}
        for policy in ({}, {"allow_network": True}, {"allow_paid": True}):
            with self.assertRaises(ValueError):
                registry.build("relation_predictor", spec, policy)
            self.assertEqual(constructed, [])
        registry.build("relation_predictor", spec, {"allow_network": True, "allow_paid": True})
        self.assertEqual(constructed, [True])


if __name__ == "__main__":
    unittest.main()
