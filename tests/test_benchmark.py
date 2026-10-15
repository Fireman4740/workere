"""Offline regression tests. Synthetic fixtures do NOT prove historical run parity."""
from __future__ import annotations

import copy
import json
import random
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

from benchmark import data, labels, report, voting
from benchmark.evaluate import compare, evaluate, evaluate_event_mentions
from benchmark.examples.external import predict
from benchmark.examples.openrouter_decisions import MODELS, decode, request_body
from benchmark.history import import_history, verify, verify_many
from benchmark.io import read_json, write_json, write_jsonl
from benchmark.suite import documents, freeze, model_inputs

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "a253e5e"  # The audited dev commit, not a moving branch.


def config():
    return {"active_dataset": "synthetic", "seed": 42,
            "datasets": {"synthetic": {"name": "synthetic-only", "repo_id": "local/synthetic",
                "split": "train", "text_field": "text", "ann_field": "annots", "max_examples": 3,
                "kfold": {"enabled": True, "n_folds": 2}, "constrain_to_pair_list": True}},
            "data": {"binary_undirected": False, "vote_to_binary": False,
                     "binary_default": "norel", "novote_norel": True,
                     "labels": ["causes", "causedby", "norel"], "shuffle_pair_list": True,
                     "shuffle_docs": True, "reshuffle_per_resample": True},
            "experiment": {"resampling": {"enabled": True, "n_runs": 3, "tie_breaking": "norel"}}}


def rows():
    return [{"id": f"d{i}", "text": "<a rain> caused <b flooding> . <c evacuation>",
             "annots": "<a rain> causes <b flooding>, <b flooding> causedby <a rain>",
             "tokens": ["rain", "caused", "flooding", ".", "evacuation"],
             "mentions": ["a", "b", "c"], "spans": [[0], [2], [4]],
             "sentences": [[0, 4], [4, 5]], "lang": "eng" if i % 2 else "fra",
             **({"pair_list": [[0, 1], [1, 0], [0, 2]]} if i % 2 else {})} for i in range(4)]


def predictions(suite):
    return [{"id": d.id, "triples": [["a", "causes", "b"], ["b", "causedby", "a"]]}
            for d in documents(suite)]


def original_modules():
    """Load immutable pre-migration source into a separate namespace for an independent oracle."""
    namespace = "_benchmark_dev_reference"
    package = types.ModuleType(namespace)
    sys.modules[namespace] = package
    for name in ("labels", "metrics", "data", "voting", "report"):
        source = subprocess.check_output(["git", "show", f"{BASELINE}:eci/{name}.py"],
                                         cwd=ROOT, text=True)
        source = source.replace("from eci.labels import", f"from {namespace}.labels import")
        source = source.replace("from eci import metrics as M", f"from {namespace} import metrics as M")
        module = types.ModuleType(f"{namespace}.{name}")
        sys.modules[module.__name__] = module
        setattr(package, name, module)
        exec(compile(source, f"{BASELINE}:eci/{name}.py", "exec"), module.__dict__)
    return package


class Benchmark(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.suite = freeze(config(), rows())

    def test_freeze_sampling_candidates_and_folds_match_original(self):
        old = original_modules()
        cfg = config()
        expected = [old.data.doc_from_row(row, i, text_field="text", ann_field="annots",
                    valid_labels=set(cfg["data"]["labels"]), binary_undirected=False,
                    shuffle_pair_list=True) for i, row in enumerate(rows())]
        expected = old.data.sample(expected, 3, True, 42)
        actual = documents(self.suite)
        self.assertEqual([d.id for d in expected], [d.id for d in actual])
        for a, b in zip(actual, expected):
            self.assertEqual([tuple(x) for x in a.pair_list], b.pair_list)
            self.assertEqual([tuple(x) for x in a.gold], b.gold)
            self.assertEqual(a.fold, b.idx % 2)
        cfg["datasets"]["synthetic"]["max_examples"] = 4
        self.assertEqual([d.id for d in actual], [d.id for d in documents(freeze(cfg, rows()))][:3])

    def test_extracted_core_is_original_source_apart_from_imports(self):
        for name in ("labels", "metrics", "data", "voting", "report"):
            original = subprocess.check_output(["git", "show", f"{BASELINE}:eci/{name}.py"],
                                               cwd=ROOT, text=True)
            migrated = (ROOT / "benchmark" / f"{name}.py").read_text()
            self.assertEqual(migrated.replace("from benchmark", "from eci"), original, name)

    def test_scores_match_independent_original_across_decision_rules(self):
        old = original_modules()
        for mode in ("directed", "binary_undirected", "vote_to_binary"):
            for tie in ("norel", "random"):
                for seed in range(8):
                    cfg = config()
                    cfg["seed"] = seed
                    cfg["experiment"]["resampling"]["tie_breaking"] = tie
                    if mode != "directed":
                        cfg["data"][mode] = True
                    suite = freeze(cfg, rows())
                    docs = documents(suite)
                    rng = random.Random(seed)
                    all_predictions, expected = [], []
                    for d in docs:
                        passes = [[(rng.choice(["a", "b", "ghost"]),
                                    rng.choice(["causes", "causedby", "norel", "none", "cuses"]),
                                    rng.choice(["b", "c"])) for _ in range(rng.randrange(6))]
                                  for _ in range(3)]
                        all_predictions.append({"id": d.id, "passes": passes})
                        final, stats = old.voting.aggregate(passes, tie,
                                           random.Random(f"{seed}::{d.id}::vote"))
                        if mode == "binary_undirected":
                            final = [(min(s, t), lab, max(s, t)) for s, lab, t in final]
                        gold = d.gold
                        if mode == "vote_to_binary":
                            final = old.voting.votes_to_binary(stats, "norel", True)
                            gold = [t for t in old.data.to_binary_undirected(gold) if t[1] != "norel"]
                        expected.append({"id": d.id, "doc_idx": d.idx, "lang": d.lang,
                            "gold_triples": gold, "pred_triples": final, "pair_stats": stats,
                            "mention_sentence": d.sentence_of, "first_token": d.first_token,
                            "surfaces": d.surfaces, "failure": None, "justifications": {}})
                    ls = labels.DIRECTED_LABELS if mode == "directed" else labels.BINARY_LABELS
                    baseline = old.report.build(expected, len(docs), cfg, ls)
                    baseline["kfold_n_folds"] = 2
                    actual = evaluate(suite, all_predictions, representation="passes")["results"]
                    self.assertEqual(actual, baseline, (mode, tie, seed))

    def test_synthetic_replay_import_and_corruption_gate(self):
        result = evaluate(self.suite, predictions(self.suite))
        path = self.path / "synthetic_report.json"
        write_json(path, result)
        self.assertEqual(verify(path)["status"], "equal")
        imported = import_history(self.suite, path, model="synthetic-fixture")
        self.assertEqual(imported["historical_source"]["parity"], "equal")
        self.assertEqual(len(compare([result, imported])), 2)
        changed = copy.deepcopy(result)
        changed["results"]["micro_f1"] = .123
        bad = self.path / "corrupted.json"
        write_json(bad, changed)
        self.assertEqual(verify(bad)["status"], "different")
        with self.assertRaisesRegex(ValueError, "parity failed"):
            import_history(self.suite, bad, model="bad")
        with self.assertRaisesRegex(ValueError, "NOT demonstrated"):
            verify_many([])
        incomplete = copy.deepcopy(result)
        del incomplete["results"]["binary"]
        missing = self.path / "incomplete.json"
        write_json(missing, incomplete)
        with self.assertRaisesRegex(ValueError, "headline metrics"):
            import_history(self.suite, missing, model="invalid")

    def test_history_import_rejects_changed_gold_with_identical_headline_scores(self):
        result = evaluate(self.suite, [{"id": d.id, "triples": []} for d in documents(self.suite)])
        path = self.path / "synthetic.json"
        write_json(path, result)
        # Swap the meaning of the two gold labels; empty predictions keep all headline F1s zero.
        changed = copy.deepcopy(self.suite)
        for d in changed["documents"]:
            d["gold"] = [[s, "causedby" if lab == "causes" else "causes", t]
                         for s, lab, t in d["gold"]]
        from benchmark.io import fingerprint
        changed["fingerprint"] = fingerprint({k: v for k, v in changed.items() if k != "fingerprint"})
        with self.assertRaisesRegex(ValueError, "parity failed"):
            import_history(changed, path, model="wrong-gold")

    def test_checkpoint_rebuild_and_corruption(self):
        suite, preds = self.suite, predictions(self.suite)
        result = evaluate(suite, preds)
        records = []
        for d, p in zip(documents(suite), preds):
            records.append({"id": d.id, "doc_idx": d.idx, "lang": d.lang,
                "gold_triples": d.gold, "pred_triples": p["triples"],
                "pair_stats": voting.single_pass_stats(p["triples"]),
                "mention_sentence": d.sentence_of, "first_token": d.first_token,
                "surfaces": d.surfaces, "justifications": {}, "failure": None})
        path = self.path / "synthetic.json"
        write_json(path, result)
        write_jsonl(self.path / "docs.jsonl", records)
        replay = verify(path)
        self.assertEqual(replay["level"], "checkpoint-to-report")
        self.assertEqual(replay["status"], "equal")
        records[0]["pred_triples"] = []
        bad = self.path / "bad_docs.jsonl"
        write_jsonl(bad, records)
        self.assertEqual(verify(path, bad)["status"], "different")

    def test_report_replay_includes_documents_with_no_scored_pairs(self):
        cfg, raw = config(), rows()
        raw[0]["annots"] = ""
        cfg["datasets"]["synthetic"]["max_examples"] = 0
        suite = freeze(cfg, raw)
        preds = [{"id": d.id, "triples": []} for d in documents(suite)]
        result = evaluate(suite, preds)
        path = self.path / "empty-doc.json"
        write_json(path, result)
        self.assertEqual(verify(path)["status"], "equal")

    def test_historical_unknown_scopes_and_offspec_labels_are_preserved(self):
        preds = [{"id": d.id, "triples": [["ghost", "causes,maybe", "a"]]}
                 for d in documents(self.suite)]
        result = evaluate(self.suite, preds)
        self.assertIn("causes,maybe", result["results"]["per_label"])
        self.assertEqual(result["results"]["sentence_unknown_pairs"], len(preds))
        path = self.path / "offspec.json"
        write_json(path, result)
        self.assertEqual(verify(path)["status"], "equal")

    def test_coverage_format_and_suite_tampering_rejected(self):
        preds = predictions(self.suite)
        for bad in (preds[:-1], preds + [preds[0]], preds + [{"id": "extra", "triples": []}],
                    [{**p, "triples": [["a", "causes"]]} for p in preds],
                    [{"id": p["id"], "passes": []} for p in preds]):
            with self.assertRaises(ValueError):
                evaluate(self.suite, bad)
        changed = copy.deepcopy(self.suite)
        changed["documents"][0]["gold"] = []
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            evaluate(changed, preds)
        with self.assertRaisesRegex(ValueError, "expected 3"):
            evaluate(self.suite, [{"id": p["id"], "passes": [[]]} for p in preds], representation="passes")

    def test_legacy_imports_refer_to_same_functions(self):
        from eci import data as D, report as R, metrics as M, voting as V
        from benchmark import metrics
        self.assertIs(D.load_eval_docs, data.load_eval_docs)
        self.assertIs(R.build, report.build)
        self.assertIs(M.binary, metrics.binary)
        self.assertIs(V.aggregate, voting.aggregate)

    def test_inputs_are_gold_free_and_extraction_has_no_oracle_markup(self):
        for row in model_inputs(self.suite):
            self.assertFalse({"gold", "gold_triples", "annots"} & set(row))
        for row in model_inputs(self.suite, "event_mentions"):
            self.assertEqual(set(row), {"id", "tokens", "sentences", "lang"})
            self.assertNotIn("<a", " ".join(row["tokens"]))

    def test_untyped_exact_event_mentions_and_invalid_span(self):
        preds = [{"id": d.id, "spans": [[0], [1]]} for d in documents(self.suite)]
        result = evaluate_event_mentions(self.suite, preds)["results"]
        self.assertEqual(result["precision"], .5)
        self.assertEqual(result["recall"], 1 / 3)
        self.assertAlmostEqual(result["f1"], .4)
        for span in ([[99]], [[]], [[True]], [[2, 0]]):
            bad = [{"id": d.id, "spans": span} for d in documents(self.suite)]
            with self.assertRaises(ValueError):
                evaluate_event_mentions(self.suite, bad)

    def test_compare_rejects_different_suites_tasks_and_representations(self):
        a = evaluate(self.suite, predictions(self.suite))
        self.assertEqual(len(compare([a, a])), 2)
        for field in ("suite_fingerprint", "task", "protocol", "representation"):
            b = copy.deepcopy(a)
            b[field] = "different"
            with self.assertRaises(ValueError):
                compare([a, b])

    def test_scored_pairs_are_union_not_all_candidates(self):
        preds = [{"id": d.id, "triples": []} for d in documents(self.suite)]
        result = evaluate(self.suite, preds)["results"]
        self.assertEqual(result["total_pairs"], 6)  # Only the two directed gold pairs per doc.
        self.assertEqual(result["binary"]["f1"], 0)
        preds[0]["triples"].append(["ghost", "cuses", "a"])
        result = evaluate(self.suite, preds)["results"]
        self.assertEqual(result["sentence_unknown_pairs"], 1)
        self.assertEqual(result["total_pairs"], 7)

    def test_decision_providers_have_distinct_requests_and_decoding(self):
        mercury = request_body(MODELS[0], "rain causes flooding", "rain", "flooding")
        span = request_body(MODELS[1], "rain causes flooding", "rain", "flooding")
        self.assertEqual(mercury["questions"]["relation"]["type"], "choice")
        self.assertTrue(all(q["type"] == "noul" for q in span["questions"].values()))
        self.assertIsInstance(span["state"], str)
        self.assertEqual(decode(MODELS[0], {"answers": {"relation": {"choice": "B"}}}), "causedby")
        self.assertEqual(decode(MODELS[1], {"answers": {"forward": {"noul": .9},
                                                    "reverse": {"noul": .1}}}), "causes")
        self.assertEqual(decode(MODELS[1], {"answers": {"forward": {"noul": .9},
                                                    "reverse": {"noul": .9}}}), "norel")

    def test_external_adapter_and_cli_in_isolated_stdlib_python(self):
        standalone = self.path / "standalone"
        shutil.copytree(ROOT / "benchmark", standalone / "benchmark",
                        ignore=shutil.ignore_patterns("__pycache__", ".venv", "build", "dist", "*.egg-info"))
        inputs = list(model_inputs(self.suite))
        preds = [predict(row) for row in inputs]
        write_json(self.path / "suite.json", self.suite)
        write_jsonl(self.path / "preds.jsonl", preds)
        out = self.path / "out.json"
        # -S disables site packages: no datasets, yaml, langchain, eci, or project installation.
        subprocess.run([sys.executable, "-S", "-m", "benchmark", "score", "--suite",
            str(self.path / "suite.json"), "--predictions", str(self.path / "preds.jsonl"),
            "--model", "toy-norel", "--output", str(out)], cwd=standalone, check=True,
            capture_output=True, text=True)
        self.assertEqual(read_json(out)["results"]["binary"]["f1"], 0)
        again = subprocess.run([sys.executable, "-S", "-m", "benchmark", "score", "--suite",
            str(self.path / "suite.json"), "--predictions", str(self.path / "preds.jsonl"),
            "--model", "toy-norel", "--output", str(out)], cwd=standalone,
            capture_output=True, text=True)
        self.assertEqual(again.returncode, 2)  # Existing artifacts cannot be overwritten.


if __name__ == "__main__":
    unittest.main()
