"""The Causal-TimeBank dataprep (dataprep/ctb_standard_eci.py) against its sources and the literature.

Needs the CTB release with CAT/ and TimeML/ extracted (litt/Causal-TimeBank, or CTB_ROOT=...);
the provenance check needs the published dataset in the HF cache. Each skips otherwise.
Run: python -m unittest tests.test_ctb_dataprep -v
"""
from __future__ import annotations

import glob
import os
import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dataprep"))
import ctb_standard_eci as ctb  # noqa: E402

CTB = Path(os.environ.get("CTB_ROOT", ROOT.parent / "litt" / "Causal-TimeBank"))


@unittest.skipUnless((CTB / "CAT").is_dir() and (CTB / "TimeML").is_dir(), f"no CTB release at {CTB}")
class CTBDataprep(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = ctb.build_rows(str(CTB))

    def test_counts_match_the_release(self):
        """README of the release: 6,811 EVENTs, 318 CLINKs, 171 C-SIGNALs (169 used by a CLINK)."""
        self.assertEqual(len(self.rows), 183)
        self.assertEqual(sum(len(r["mentions"]) for r in self.rows), 6811)
        self.assertEqual(sum(len(r["clinks"]["source"]) for r in self.rows), 318)
        self.assertEqual(sum(len(r["relations"]["causes"]) for r in self.rows), 298)
        self.assertEqual(sum(1 for r in self.rows for s in r["clinks"]["signal"] if s), 169)
        self.assertEqual(sum(len(r["pair_list"]) for r in self.rows), 2 * 9721)

    def test_layers_are_consistent(self):
        for r in self.rows:
            causes = r["relations"]["causes"]
            self.assertEqual(r["relations"]["causedby"], [[e, c] for c, e in causes])
            self.assertEqual(r["annots"].count(" causes "), len(causes))
            cl = r["clinks"]
            self.assertEqual(causes, [[s, t] for s, t, i in zip(cl["source"], cl["target"], cl["intra"]) if i])
            self.assertTrue({tuple(p) for p in causes} <= {tuple(p) for p in r["pair_list"]})
            sent = [next(k for k, (a, b) in enumerate(r["sentences"]) if a <= sp[0] < b) for sp in r["spans"]]
            for s, t, intra in zip(cl["source"], cl["target"], cl["intra"]):
                self.assertEqual(intra, sent[s] == sent[t])

    def test_direction_agrees_with_timeml(self):
        """Every CLINK as TimeML states it: eventInstanceID (cause) -> relatedToEventInstance (effect)."""
        timeml = set()
        for f in glob.glob(str(CTB / "TimeML" / "*.tml")):
            for c in ET.parse(f).getroot().findall("CLINK"):
                timeml.add((Path(f).stem, c.get("eventInstanceID"), c.get("relatedToEventInstance")))
        built = {(r["id"], r["mentions"][s], r["mentions"][t])
                 for r in self.rows for s, t in zip(r["clinks"]["source"], r["clinks"]["target"])}
        self.assertEqual(len(timeml), 318)
        self.assertEqual(built, timeml)

    def test_against_the_published_dataset(self):
        """Same documents, same 298 causal pairs and direction; 4 mentions recovered."""
        try:
            counts = ctb.check_against("Nofing/CausalTimeBank-standard-eci", self.rows)
        except AssertionError:
            raise
        except Exception as e:  # no network and no cache
            self.skipTest(f"published dataset unavailable: {e}")
        self.assertEqual(counts["causal pairs identical (ids and direction)"], 298)
        self.assertEqual(counts["mentions added"], 4)
        self.assertEqual(counts["sentences, published"], counts["sentences, CAT"])


if __name__ == "__main__":
    unittest.main()
