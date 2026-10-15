"""Independent protocols, gold isolation, components, and sampling uncertainty."""
import copy
import tempfile
import unittest
from pathlib import Path

from benchmark.evaluate import evaluate
from benchmark.interfaces import Context
from benchmark.io import write_json, write_jsonl
from benchmark.pipeline import build_pipeline
from benchmark.protocols import aggregate_statistics, public_inputs, score_predictions
from benchmark.registry import load_plugins
from benchmark.schema import Entity, Prediction, Relation
from benchmark.statistics import document_bootstrap, run_variability
from benchmark.suite import documents, freeze
from tests.test_benchmark import config, rows


def component(name, **params):
    return {"plugin": name, "version": "1", "params": params}


def heuristic(protocol):
    pipeline = {"entity_extractor": component("reference") if protocol == "relations_gold" else
                component("lexicon", terms=["rain", "flooding", "evacuation"])}
    if protocol != "entity_extraction":
        pipeline.update(candidate_generator=component("historical" if protocol == "relations_gold" else "all_ordered"),
                        relation_predictor=component("heuristic"))
    return pipeline


class Protocols(unittest.TestCase):
    def setUp(self):
        self.suite = freeze(config(), rows())

    def produce(self, protocol, pipeline=None):
        pipeline, _ = build_pipeline(pipeline or heuristic(protocol), protocol, load_plugins(), {})
        out = []
        for inputs in public_inputs(self.suite, protocol):
            ctx = Context(42, protocol, inputs.document.id)
            out.append(pipeline.predict(inputs, ctx).to_dict(inputs.document))
        pipeline.close()
        return out

    def test_heuristic_complete_pipeline_all_three_protocols(self):
        for protocol in ("entity_extraction", "relations_gold", "end_to_end"):
            result = score_predictions(self.suite, self.produce(protocol), protocol)
            self.assertEqual(result["summary"]["primary"]["f1"], 1)
            self.assertEqual(result["oracle_reference_mentions"], protocol == "relations_gold")
            aggregate = aggregate_statistics(result["document_statistics"])
            for name, values in result["summary"].items():
                for key in ("precision", "recall", "f1"):
                    self.assertAlmostEqual(aggregate[name][key], values[key], places=14)

    def test_gold_relation_scoring_is_exactly_the_existing_scorer(self):
        preds = self.produce("relations_gold")
        triples = [{"id": row["id"], "triples": [[r["source"], r["label"], r["target"]]
                   for r in row["relations"]]} for row in preds]
        expected = evaluate(self.suite, triples)["results"]
        actual = score_predictions(self.suite, preds, "relations_gold")
        self.assertEqual(actual["results"], expected)

    def test_models_get_no_relational_gold_or_oracle_mentions_in_end_to_end(self):
        for protocol in ("end_to_end", "entity_extraction"):
            for inputs in public_inputs(self.suite, protocol):
                self.assertIsNone(inputs.reference_entities)
                self.assertIsNone(inputs.reference_candidates)
                self.assertNotIn("<a", inputs.document.text)
                self.assertNotIn("gold", inputs.to_dict())
        for inputs in public_inputs(self.suite, "relations_gold"):
            self.assertIsNotNone(inputs.reference_entities)
            self.assertFalse(hasattr(inputs, "gold"))
            self.assertFalse(hasattr(inputs.document, "gold"))

    def test_reference_extractor_refuses_nonoracle_protocol(self):
        pipeline = {"entity_extractor": component("reference")}
        built, _ = build_pipeline(pipeline, "entity_extraction", load_plugins(), {})
        inputs = next(public_inputs(self.suite, "entity_extraction"))
        with self.assertRaisesRegex(ValueError, "oracle"):
            built.predict(inputs, Context(42, "entity_extraction", inputs.document.id))

    def test_missing_and_hallucinated_spans_reduce_end_to_end_scores(self):
        pred = self.produce("end_to_end")
        inputs = {i.document.id: i for i in public_inputs(self.suite, "end_to_end")}
        for row in pred:
            # Correct relation IDs but one endpoint span shifted: entity and relation scores diverge.
            row["entities"][0]["token_indices"] = [1]
            row["entities"][0].pop("char_offsets")
        result = score_predictions(self.suite, pred, "end_to_end")
        self.assertEqual(result["summary"]["primary"]["f1"], 0)
        self.assertAlmostEqual(result["summary"]["entities"]["f1"], 2 / 3)
        for row in pred:
            row["relations"].append({"source": "unresolved", "target": "p1", "label": "causes"})
        result = score_predictions(self.suite, pred, "end_to_end")
        self.assertEqual(result["results"]["sentence_unknown_relations"], len(pred))

    def test_relation_confidence_is_not_a_scoring_threshold(self):
        high = self.produce("relations_gold")
        low = copy.deepcopy(high)
        for row in low:
            for relation in row["relations"]:
                relation["confidence"] = .001
        self.assertEqual(score_predictions(self.suite, high, "relations_gold")["summary"],
                         score_predictions(self.suite, low, "relations_gold")["summary"])

    def test_postprocessor_and_reference_ids_validation(self):
        cfg = heuristic("relations_gold")
        cfg["postprocessors"] = [component("relation_threshold", threshold=.95)]
        result = score_predictions(self.suite, self.produce("relations_gold", cfg), "relations_gold")
        self.assertEqual(result["summary"]["primary"]["f1"], 0)
        pred = self.produce("relations_gold")
        pred[0]["entities"][0]["id"] = "other"
        with self.assertRaisesRegex(ValueError, "preserve"):
            score_predictions(self.suite, pred, "relations_gold")

    def test_monolithic_saved_and_legacy_adapters(self):
        pred = self.produce("relations_gold")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.jsonl"
            write_jsonl(path, pred)
            adapted = self.produce("relations_gold", {"monolithic": component("saved_predictions", path=str(path))})
            self.assertEqual(adapted, pred)
            result = score_predictions(self.suite, pred, "relations_gold")["results"]
            report = Path(tmp) / "old-report.json"
            write_json(report, {"config": self.suite["config"], "results": result})
            replay = self.produce("relations_gold", {"monolithic": component("legacy_report", path=str(report))})
            self.assertEqual(score_predictions(self.suite, replay, "relations_gold")["results"]["micro_f1"], 1)

    def test_external_jev_plugin_predicts_from_recorded_probabilities(self):
        load_plugins(["benchmark.examples.jev_plugin"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "decisions.jsonl"
            records = [{"id": d.id, "source": a, "target": b,
                        "probabilities": {"A": .9, "B": .05, "C": .05}}
                       for d in documents(self.suite) for a, b in d.pair_list]
            write_jsonl(path, records)
            cfg = heuristic("relations_gold")
            cfg["relation_predictor"] = {"plugin": "jev_recorded", "version": "example-1", "params": {"path": str(path)}}
            pred = self.produce("relations_gold", cfg)
            self.assertTrue(all(r["confidence"] == .9 for row in pred for r in row["relations"]))


class Uncertainty(unittest.TestCase):
    def test_document_bootstrap_deterministic_and_counts_weight_repeated_documents(self):
        records = [{"id": "good", "metrics": {"primary": {"tp": 1, "fp": 0, "fn": 0}}},
                   {"id": "bad", "metrics": {"primary": {"tp": 0, "fp": 1, "fn": 1}}}]
        self.assertAlmostEqual(aggregate_statistics([records[0], records[0], records[1]])["primary"]["f1"], 2 / 3)
        a = document_bootstrap(records, samples=100, seed=8)
        self.assertEqual(a, document_bootstrap(records, samples=100, seed=8))
        self.assertEqual(a["intervals"]["primary"]["f1"], [0, 1])

    def test_seed_variance_is_distinct_from_document_uncertainty(self):
        self.assertEqual(run_variability([0, 1])["sample_variance"], .5)
        self.assertEqual(run_variability([0, 1])["mean"], .5)
        self.assertIsNone(run_variability([.5])["sample_variance"])
        self.assertIsNone(run_variability([.5])["mean_confidence_interval"])
        for kwargs in ({"samples": -1}, {"confidence": 1}, {"confidence": True}):
            with self.assertRaises(ValueError):
                run_variability([.5], **kwargs)


if __name__ == "__main__":
    unittest.main()
