"""Scientific data and offline artifact regression checks (stdlib only)."""
import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE=Path(__file__).resolve().parent


class SiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=json.loads((HERE/"data.json").read_text())
        cls.ex=cls.data["experiments"]
        cls.by_n={e["number"]:e for e in cls.ex}

    def test_complete_snapshot(self):
        self.assertEqual(list(range(1,118)),[e["number"] for e in self.ex])
        self.assertEqual(len({e["id"] for e in self.ex}),117)
        runs=[r for e in self.ex for r in e["runs"]]
        self.assertEqual(len(runs),190)
        self.assertEqual(len({r["id"] for r in runs}),190)
        self.assertEqual(sum(r["log_available"] for r in runs),187)

    def test_each_experiment_has_its_own_method_and_full_detail(self):
        self.assertEqual(len({e["diagram"]["intervention"] for e in self.ex}),117)
        for e in self.ex:
            for key in ["method","evaluation","outcome","checks","technical","simple"]:
                self.assertTrue(e[key],(e["number"],key))
            self.assertGreater(len(e["technical"]),600)

    def test_prompts_have_frozen_sources(self):
        bank=self.data['prompt_bank']
        self.assertGreater(len(bank),100)
        for e in self.ex:
            self.assertTrue(e['prompt_status'])
            if not e['prompt_refs']:
                self.assertTrue(e.get('prompt_upstream') or e['state']=='planifie',e['number'])
            for ref in e['prompt_refs']:
                self.assertIn(ref['id'],bank)
                self.assertEqual(len(ref['commit']),40)
                self.assertEqual(len(ref['blob_sha256']),64)
                self.assertTrue(bank[ref['id']]['content'])

    def test_f1_and_ratios_recompute_from_counts(self):
        checked=0
        for e in self.ex:
            for r in e["rows"]:
                self.assertTrue(math.isfinite(r["f1"]))
                self.assertTrue(0<=r["f1"]<=100)
                if r["tp"] is not None:
                    checked+=1
                    tp,fp,fn=r["tp"],r["fp"],r["fn"]
                    self.assertLessEqual(abs(r["f1"]-(200*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)),.016,(e["number"],r["variant"]))
                    for field,num,denom in [("precision",tp,tp+fp),("recall",tp,tp+fn)]:
                        if denom and r[field] is not None:self.assertLessEqual(abs(r[field]-100*num/denom),.016)
        self.assertGreater(checked,700)

    def test_rows_unique_and_have_sources(self):
        for e in self.ex:
            keys=[(r["context"],r["variant"]) for r in e["rows"]]
            self.assertEqual(len(keys),len(set(keys)))
            for r in e["rows"]:
                self.assertTrue(r["source"]["file"])
                self.assertTrue(r["source"]["pointer"])
                if r["ci"]:
                    self.assertEqual(len(r["ci"]),2)
                    self.assertLessEqual(r["ci"][0],r["ci"][1])
                    self.assertIsNotNone(r["delta"])

    def test_w4_full_matrix_w5_w6_effective_scope(self):
        for n,count,docs in [(113,96,47),(114,96,4),(115,24,4)]:
            e=self.by_n[n]
            self.assertEqual(len(e["rows"]),count)
            self.assertEqual({r["documents"] for r in e["rows"]},{docs})
        observations=self.by_n[115]["observations"]
        self.assertEqual({o["label"]:o["value"] for o in observations},{"base · lectures valides":4,"cc · lectures valides":11})

    def test_replay_corrections(self):
        rows={r["variant"]:r for r in self.by_n[74]["rows"]}
        self.assertAlmostEqual(rows["E1/temp0.6"]["f1"],44.03669724770642)
        self.assertEqual([rows["E1/control"][k] for k in ["tp","fp","fn"]],[42,26,98])
        rows={r["variant"]:r for r in self.by_n[75]["rows"]}
        self.assertNotEqual(rows["E2/control"]["f1"],rows["E2/early_exit"]["f1"])
        row=next(r for r in self.by_n[84]["rows"] if r["variant"]=="P1/verify_b8k")
        self.assertEqual([row[k] for k in ["tp","fp","fn"]],[74,48,66])
        self.assertAlmostEqual(row["f1"],56.48854961832061)
        self.assertIsNone(row["ci"])

    def test_j3_corrected_gold_only(self):
        self.assertEqual(sorted(r["f1"] for r in self.by_n[29]["rows"]),[36.07,36.35])
        self.assertEqual(sorted(r["f1"] for r in self.by_n[30]["rows"]),[35.99,36.07])
        for n in [29,30]:self.assertTrue(all(r["source"]["pointer"]!="CONFIRM_RESULT" for r in self.by_n[n]["rows"]))

    def test_incomplete_and_planned_do_not_gain_fake_f1(self):
        for n in [6,13,26,40,43,45,47,49,54,55,63,64,65,92,106,107,108,116,117]:
            self.assertEqual(self.by_n[n]["rows"],[],n)
        self.assertEqual(sorted(o["value"] for o in self.by_n[63]["observations"]),[63,68,69])

    def test_h1_unpaired_history_separate(self):
        self.assertFalse(any(r["variant"]=="historical_unpaired_reference" for r in self.by_n[39]["rows"]))
        self.assertEqual(len(self.by_n[39]["rows"]),6)

    def test_zero_and_absence_are_distinct(self):
        self.assertEqual(self.by_n[1]["rows"][0]["f1"],0)
        self.assertIsNone(self.by_n[18]["rows"][0]["tp"])
        self.assertEqual(self.by_n[116]["rows"],[])

    def test_static_artifact_self_contained(self):
        html=(HERE/"index.html").read_text()
        self.assertNotRegex(html,r'<script[^>]+src=')
        self.assertNotRegex(html,r'<link[^>]+(?:stylesheet|preload)')
        self.assertNotRegex(html,r'\b(?:fetch|XMLHttpRequest|WebSocket)\s*\(')
        payload=re.search(r'<script id="research-data" type="application/json">(.*?)</script>',html,re.S)[1]
        self.assertEqual(json.loads(payload),self.data)

    def test_build_reproducible_without_original_artifacts(self):
        with tempfile.TemporaryDirectory() as td:
            output=Path(td)/"offline.html"
            subprocess.run([sys.executable,str(HERE/"build_site.py"),"--output",str(output)],check=True,capture_output=True)
            self.assertEqual(hashlib.sha256(output.read_bytes()).digest(),hashlib.sha256((HERE/"index.html").read_bytes()).digest())


if __name__=="__main__": unittest.main(verbosity=2)
