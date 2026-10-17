import json
import tempfile
import unittest
from pathlib import Path

from pipelines.jev.llm import frozen_demonstration_bank
from pipelines.jev.round4_eval import evaluate
from benchmark.configuration import expand_campaign
from benchmark.suite import documents
from benchmark.configuration import load_suite


class Round4Contracts(unittest.TestCase):
    def test_committed_screen_is_complete_and_excludes_demo_and_reserve(self):
        selection = json.loads(Path('experiments/jev/round4/selection.json').read_text())
        configs = list(expand_campaign('experiments/jev/active.yaml'))
        ids = [documents(load_suite(c))[0].id for c in configs]
        self.assertEqual(ids, selection['document_ids'])
        self.assertEqual(len(set(ids)), 44)
        self.assertEqual(len({d.split('_')[0] for d in ids}), 22)
        self.assertFalse(set(ids) & set(selection['demo_bank_ids']))
        for c in configs:
            p = c.data['pipeline']['relation_predictor']['params']
            self.assertEqual(p['passes'], 7)
            self.assertEqual(p['client']['shared_budget']['limit_usd'], 3)

    def test_frozen_bank_refuses_wrong_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'bank.json'
            p.write_text(json.dumps({'schema': 'workere-frozen-demo-bank/v1', 'repo_id': 'repo',
                                    'revision': 'old', 'documents': []}))
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                frozen_demonstration_bank(p, 'repo', 'new')

    def test_paired_fusion_uses_frozen_controls_and_rejects_missing_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controls = [dict(doc='1_example', a=a, b=b, gold=True, d1=.6, luna=0.)
                        for a, b in [('1', '2'), ('2', '1')]]
            path = root / 'controls.json'; path.write_text(json.dumps(controls))
            def prediction(complete):
                scores = [dict(source=r['a'], target=r['b'], positive_probability=1.) for r in controls[:2 if complete else 1]]
                return {'id': '1_example', 'metadata': {'decision_scores': scores,
                        'pass_triples': [[['1', 'causes', '2'], ['2', 'causedby', '1']]] * 7}}
            pred = root / 'predictions.jsonl'
            pred.write_text(json.dumps(prediction(True)) + '\n')
            receipt = {'runs': [{'status': 'completed', 'run_dir': str(root)}]}
            result = evaluate(receipt, path, 'fixture')
            self.assertEqual(result['control_fusion']['F1'], 0.)
            self.assertEqual(result['new_fusion']['F1'], 100.)
            self.assertEqual(result['primary_paired_document']['delta'], 100.)
            self.assertEqual(result['secondary_pass_ablation_no_refit']['3']['mean_fusion_F1'], 100.)
            with self.assertRaisesRegex(ValueError, 'frozen'):
                evaluate(receipt, path, 'fixture', .5)
            pred.write_text(json.dumps(prediction(False)) + '\n')
            with self.assertRaisesRegex(ValueError, 'coverage mismatch'):
                evaluate(receipt, path, 'fixture')


if __name__ == '__main__':
    unittest.main()
