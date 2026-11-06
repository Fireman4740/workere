"""CPU checks of the evaluation boundaries and threshold decision."""
import json
from pathlib import Path
import tempfile
import unittest

from ctb import load_suite, unordered_pairs
from pair_verifier import best_threshold, cached_votes, make_messages


class PairVerifierTest(unittest.TestCase):
    def test_exact_threshold_includes_lower_positive_when_f1_improves(self):
        rows = [{"label":1}, {"label":0}, {"label":1}]
        self.assertEqual(best_threshold(rows, [3,2,1]), (1, .8))

    def test_empty_final_never_uses_reasoning(self):
        suite = load_suite()
        doc = next(d for d in suite if unordered_pairs(suite[d]))
        a,b = unordered_pairs(suite[doc])[0]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"gen.jsonl"
            path.write_text("".join(json.dumps(dict(doc_id=doc, condition="x", **{"pass":i},
                finish_reason="stop", text=f'<think>[["{a}","causes","{b}"]]</think>[]'))+"\n"
                for i in range(5)))
            self.assertTrue(all(v==0 for v in cached_votes([path],suite,[doc])[doc].values()))

    def test_label_permutation_changes_demo_and_question_consistently(self):
        demo = dict(pair=("e1","e2"),surfaces=("a","b"),text="example",label=True)
        first = make_messages("rules","target",("t1","t2"),("x","y"),[demo])
        second = make_messages("rules","target",("t1","t2"),("x","y"),[demo],True)
        self.assertEqual(first[-1],second[-1])
        self.assertEqual(first[2]["content"],"A")
        self.assertEqual(second[2]["content"],"B")


if __name__ == "__main__": unittest.main()
