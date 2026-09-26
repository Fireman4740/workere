"""The MAVEN-ERE dataprep (dataprep/maven_ere_standard_eci.py) against its source and the official code.

Needs MAVEN-ERE's train.jsonl / valid.jsonl (litt/MAVEN-ERE/data/MAVEN_ERE, or MAVEN_ERE_DIR=...);
the provenance check needs the published dataset in the HF cache. Each skips otherwise.
Run: python -m unittest tests.test_maven_dataprep -v
"""
from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dataprep"))
import maven_ere_standard_eci as maven  # noqa: E402

DATA = Path(os.environ.get("MAVEN_ERE_DIR", ROOT.parent / "litt" / "MAVEN-ERE" / "data" / "MAVEN_ERE"))


@unittest.skipUnless((DATA / "valid.jsonl").exists(), f"no MAVEN-ERE data at {DATA}")
class MavenDataprep(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.splits = maven.build_splits(str(DATA))

    def test_counts(self):
        self.assertEqual({k: len(v) for k, v in self.splits.items()}, {"train": 2621, "dev": 292, "test": 710})
        test = self.splits["test"]
        self.assertEqual(sum(len(r["mentions"]) for r in test), 17780)
        self.assertEqual(sum(len(r["relations"]["causes"]) for r in test), 13624)

    def test_layers_are_consistent(self):
        for r in (r for rows in self.splits.values() for r in rows):
            causes = r["relations"]["causes"]
            self.assertEqual(r["relations"]["causedby"], [[e, c] for c, e in causes])
            self.assertEqual(len(r["causal_type"]), len(causes))
            self.assertEqual(r["annots"].count(" causes "), len(causes))
            self.assertEqual(len(r["mention_type"]), len(r["mentions"]))
            self.assertEqual(len(r["mention_source_id"]), len(r["mentions"]))
            members = [i for c in r["coref_clusters"] for i in c]
            self.assertEqual(len(members), len(set(members)))            # chains are disjoint

    def test_agrees_with_the_official_baseline(self):
        """MAVEN-ERE causal/src/data.py labels mention pairs by expanding each event pair over
        both chains; the build must carry exactly those pairs and types (valid.jsonl)."""
        with open(DATA / "valid.jsonl", encoding="utf-8") as f:
            docs = {d["id"]: d for d in map(json.loads, f)}
        for r in self.splits["test"]:
            d = docs[r["id"]]
            eid2mentions = {e["id"]: [m["id"] for m in e["mention"]] for e in d["events"]}
            official = {(a, b, rel) for rel, pairs in d["causal_relations"].items() for x, y in pairs
                        for a in eid2mentions[x] for b in eid2mentions[y]}
            ids = r["mention_source_id"]
            built = {(ids[c], ids[e], t) for (c, e), t in zip(r["relations"]["causes"], r["causal_type"])}
            self.assertEqual(built, official, r["id"])

    def test_provenance_of_the_published_dataset(self):
        try:
            counts = maven.check_against("Nofing/MAVEN-ERE-standard-eci", self.splits)
        except AssertionError:
            raise
        except Exception as e:  # no network and no cache
            self.skipTest(f"published dataset unavailable: {e}")
        self.assertEqual(sum(counts.values()), 2621 + 292 + 710)


if __name__ == "__main__":
    unittest.main()
