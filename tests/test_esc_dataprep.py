"""The ESC dataprep (dataprep/esc_standard_eci.py) against its sources and the literature.

Needs the ESC corpus (AGENTERE/data/EventStoryLine, or ESL_ROOT=...); the literature checks need
litt/{Dr.ECI,ChatGPT4CausalReasoning,PPAT}, the provenance check the published dataset in the HF
cache. Each skips otherwise.
Run: python -m unittest tests.test_esc_dataprep -v
"""
from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dataprep"))
import esc_standard_eci as esc  # noqa: E402

ESL = Path(os.environ.get("ESL_ROOT", ROOT.parent / "agentere" / "data" / "EventStoryLine"))
DRECI = ROOT.parent / "litt" / "Dr.ECI" / "data" / "ESC" / "intra_sent_causality.json"
CHATGPT4 = ROOT.parent / "litt" / "ChatGPT4CausalReasoning" / "data" / "gpt4_data" / "ESC_ECI.sample1000.jsonl"
PPAT = ROOT.parent / "litt" / "PPAT" / "data"


@unittest.skipUnless((ESL / "annotated_data" / "v0.9").is_dir(), f"no ESC corpus at {ESL}")
class ESCDataprep(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = esc.build_rows(str(ESL))
        cls.by_id = {r["id"]: r for r in cls.rows}

    def test_counts_match_the_literature(self):
        self.assertEqual(len(self.rows), 258)
        self.assertEqual(sum(r["split"] == "train" for r in self.rows), 233)
        self.assertEqual({r["topic_id"] for r in self.rows if r["split"] == "dev"}, {37, 41})
        self.assertEqual(sum(len(r["mentions"]) for r in self.rows), 5334)
        self.assertEqual(sum(len(r["relations"]["causes"]) for r in self.rows), 5602)

    def test_layers_are_consistent(self):
        for r in self.rows:
            causes = r["relations"]["causes"]
            self.assertEqual(r["relations"]["causedby"], [[e, c] for c, e in causes])
            self.assertEqual(len(r["causal_origin"]), len(causes))
            self.assertEqual(len(r["causal_frame"]), len(causes))
            self.assertEqual(len(r["coref_instance"]), len(r["mentions"]))
            self.assertEqual(r["annots"].count(" causes "), len(causes))
            self.assertEqual(len({frozenset(p) for p in causes}), len(causes))    # no pair twice

    @unittest.skipUnless(DRECI.exists(), "litt/Dr.ECI not present")
    def test_direction_agrees_with_dreci_gold(self):
        """Dr.ECI's released ESC gold: label 1 = e1 -> e2, label 2 = e2 -> e1."""
        agree = total = 0
        with open(DRECI, encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            r = json.loads(line)
            if r["tri_causal_label"] == 0:
                continue
            row = self.by_id[r["doc_name"].split("/")[-1][: -len(".xml")]]
            m = row["mentions"]
            edges = {(m[c], m[e]) for c, e in row["relations"]["causes"]}
            e1, e2 = r["events_id"]
            total += 1
            agree += ((e1, e2) if r["tri_causal_label"] == 1 else (e2, e1)) in edges
        self.assertEqual((agree, total), (3184, 3184))

    def test_gao_candidate_set(self):
        """Gao et al. (2019): without aspectual, causative, perception and reporting mentions, 639
        are dropped and 7,805 intra-sentence pairs remain; always-causal recall 98.6% (the paper's OP)."""
        mentions = intra = reachable = gold = 0
        for r in self.rows:
            keep = [esc.gao_candidate(t) for t in r["mention_type"]]
            sent = [next(k for k, (a, b) in enumerate(r["sentences"]) if a <= sp[0] < b) for sp in r["spans"]]
            mentions += sum(keep)
            by = {}
            for i, k in enumerate(keep):
                if k:
                    by[sent[i]] = by.get(sent[i], 0) + 1
            intra += sum(v * (v - 1) // 2 for v in by.values())
            for a, b in r["relations"]["causes"] + r["unlabelled_pairs"]:
                if sent[a] == sent[b]:
                    gold += 1
                    reachable += keep[a] and keep[b]
        self.assertEqual((mentions, intra, gold), (5334 - 639, 7805, 1770))
        self.assertEqual(round(100 * reachable / gold, 1), 98.6)

    @unittest.skipUnless(CHATGPT4.exists(), "litt/ChatGPT4CausalReasoning not present")
    def test_direction_agrees_with_gao_2023_sample(self):
        """Gao et al. 2023 (Is ChatGPT a Good Causal Reasoner?): 1,000 sampled ESC pairs, directed."""
        agree = 0
        with open(CHATGPT4, encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            r = json.loads(line)
            row = self.by_id[r["doc_name"].split("/")[-1][: -len(".xml")]]
            m = row["mentions"]
            edges = {(m[c], m[e]) for c, e in row["relations"]["causes"]}
            e1, e2 = r["events_id"]
            ours = 1 if (e1, e2) in edges else 2 if (e2, e1) in edges else 0
            agree += ours == r["tri_causal_label"]
        self.assertEqual(agree, len(lines))

    @unittest.skipUnless(PPAT.is_dir(), "litt/PPAT not present")
    def test_pairs_agree_with_ppat(self):
        """PPAT's released ESC folds: every gold line, the 23 `null` ones included, undirected."""
        import difflib
        pairs = docs_equal = 0
        for path in list(PPAT.glob("ESL_FOLD5/*/test.json")) + [PPAT / "ESL_dev.json"]:
            with open(path, encoding="utf-8") as f:
                docs = json.load(f)
            for d in docs:
                row = self.by_id[d["id"].split("/")[-1][: -len(".xml")]]
                sm = difflib.SequenceMatcher(None, d["tokens"], row["tokens"], autojunk=False)
                pos = {a + k: b + k for a, b, n in sm.get_matching_blocks() for k in range(n)}   # PPAT drops the URL line
                first = {sp[0]: i for i, sp in enumerate(row["spans"])}
                got = {frozenset((first[pos[x[0]]], first[pos[y[0]]])) for x, y in d["causal"]}
                want = {frozenset(p) for p in row["relations"]["causes"] + row["unlabelled_pairs"]}
                pairs += len(want)
                docs_equal += got == want
        self.assertEqual((docs_equal, pairs), (258, 5625))

    def test_annotators_explicit_causality_flags(self):
        """Where annotators flagged explicit causality, the flag and the built direction agree
        for the large majority (the raw annotation itself has a few contradictions)."""
        agree = total = 0
        for r in self.rows:
            edges = {tuple(p) for p in r["relations"]["causes"]}
            pl = r["plot_links"]
            for s, t, causes, caused_by in zip(pl["source"], pl["target"], pl["causes"], pl["caused_by"]):
                if causes == caused_by or not ({(s, t), (t, s)} & edges):
                    continue
                total += 1
                agree += ((s, t) if causes else (t, s)) in edges
        self.assertGreater(agree / total, 0.85)

    def test_provenance_of_the_published_dataset(self):
        """With the old reading of FALLING_ACTION, the builder reproduces the published dataset
        exactly — so the corrected build differs from it in direction only."""
        try:
            counts = esc.check_against("Nofing/EventStoryLine-0.9-standard-eci",
                                       esc.build_rows(str(ESL), legacy_direction=True), self.rows)
        except AssertionError:
            raise
        except Exception as e:  # no network and no cache
            self.skipTest(f"published dataset unavailable: {e}")
        self.assertEqual(counts["pairs reversed"], 2893)
        self.assertEqual(counts["pairs kept"], 2709)


if __name__ == "__main__":
    unittest.main()
