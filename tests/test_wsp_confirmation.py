"""Frozen holdout provenance, full-denominator scoring, and claim guardrails."""
from __future__ import annotations

import contextlib
import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from steering import ctb
from steering.weight_space import condition_plan, digest
from steering.wsp_pilot import sha, write_json
from steering.wsp_confirmation import (confirmation_decision, load_records,
                                       paired_comparison, score, verify_source)


def frozen_plan():
    return json.loads((Path(__file__).parents[1] / 'steering/wsp_confirmation_plan.json').read_text())


class ConfirmationTests(unittest.TestCase):
    def test_plan_keeps_all_four_randoms_and_one_candidate(self):
        p = frozen_plan()
        self.assertEqual(len(p['confirmation_documents']), 20)
        self.assertEqual(len(set(p['confirmation_documents'])), 20)
        self.assertEqual(p['selected_condition'], 'cc_0.10')
        self.assertEqual(p['conditions'], [c for c in condition_plan()
                         if c['name'] in ('base', 'cc_0.10') or 'random_seed' in c])
        self.assertEqual(p['bootstrap']['familywise_comparisons'], 5)
        # Total nominal error across five two-sided Bonferroni intervals is .05.
        lo, hi = p['bootstrap']['familywise_quantiles']
        self.assertAlmostEqual((lo + 1 - hi) * 5, .05)

    def test_source_rejects_reextraction_selection_and_document_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = frozen_plan()
            parent = {'conditions': condition_plan(), 'split': {
                'fit': ['fit'], 'screen': ['screen'], 'confirmation': plan['confirmation_documents'],
                'excluded_historical': ['old'], 'demonstrations': ['fit']}}
            parent['id'] = digest(parent)
            write_json(root / 'manifest.json', parent)
            (root / 'directions.npz').write_bytes(b'frozen vectors')
            write_json(root / 'directions.json', {'manifest_id': parent['id'],
                       'npz_sha256': sha(root / 'directions.npz')})
            write_json(root / 'screen_scores.json', {'manifest_id': parent['id'], 'complete': True})
            write_json(root / 'screen_decision.json', {'eligible_for_confirmation': ['cc']})
            plan['source_manifest_id'] = parent['id']
            plan['source_files_sha256'] = {n: sha(root / n) for n in plan['source_files_sha256']}
            self.assertEqual(verify_source(root, plan), parent)
            changed = copy.deepcopy(plan)
            changed['confirmation_documents'] = ['screen'] + changed['confirmation_documents'][1:]
            with self.assertRaisesRegex(ValueError, 'reserved split'):
                verify_source(root, changed)
            changed = copy.deepcopy(plan)
            changed['conditions'] = changed['conditions'][:-1]
            with self.assertRaisesRegex(ValueError, 'all four random'):
                verify_source(root, changed)
            changed = copy.deepcopy(plan)
            changed['selected_condition'] = 'recall_random0_0.10'
            with self.assertRaisesRegex(ValueError, 'candidate'):
                verify_source(root, changed)
            (root / 'directions.npz').write_bytes(b'refitted vectors')
            with self.assertRaisesRegex(ValueError, 'frozen source changed'):
                verify_source(root, plan)

    def test_paired_bootstrap_sign_scale_and_identical_outputs(self):
        bootstrap = {**frozen_plan()['bootstrap'], 'draws': 200}
        a = {'doc': {'tp': 1, 'fp': 1, 'fn': 2}}
        b = {'doc': {'tp': 2, 'fp': 0, 'fn': 1}}
        result = paired_comparison(a, b, bootstrap)
        self.assertEqual(result, {'delta_f1_pp': 40., 'ci95_pp': [40., 40.],
                                  'ci99_bonferroni_pp': [40., 40.]})
        self.assertEqual(paired_comparison(a, a, bootstrap)['ci99_bonferroni_pp'], [0., 0.])
        self.assertEqual(paired_comparison(b, a, bootstrap)['ci95_pp'], [-40., -40.])
        with self.assertRaises(ValueError):
            paired_comparison(a, {'other': b['doc']}, bootstrap)

    def test_primary_gain_does_not_establish_direction_specificity(self):
        plan = frozen_plan()
        results = {'base': {'precision': .8, 'invalid_answers': 0},
                   'cc_0.10': {'precision': .82, 'invalid_answers': 0}}
        comparisons = {c['name']: {'ci95_pp': [1., 3.], 'ci99_bonferroni_pp': [.1, 4.]}
                       for c in plan['conditions'] if c['name'] != 'cc_0.10'}
        comparisons['recall_random0_0.10']['ci99_bonferroni_pp'] = [-5., 2.]
        result = confirmation_decision(results, comparisons, plan)
        self.assertTrue(result['primary_task_gain_confirmed'])
        self.assertFalse(result['direction_specific_gain_confirmed'])
        comparisons['recall_random0_0.10']['ci99_bonferroni_pp'] = [.2, 2.]
        self.assertTrue(confirmation_decision(results, comparisons, plan)['direction_specific_gain_confirmed'])
        comparisons['base']['ci95_pp'] = [0., 3.]
        comparisons['base']['ci99_bonferroni_pp'] = [-1., 4.]
        self.assertFalse(confirmation_decision(results, comparisons, plan)['primary_task_gain_confirmed'])

    def test_precision_and_invalid_guardrails_block_positive_intervals(self):
        plan = frozen_plan()
        comparisons = {c['name']: {'ci95_pp': [1., 3.], 'ci99_bonferroni_pp': [.1, 4.]}
                       for c in plan['conditions'] if c['name'] != 'cc_0.10'}
        for precision, invalid in ((.77, 0), (.82, 1)):
            results = {'base': {'precision': .8, 'invalid_answers': 0},
                       'cc_0.10': {'precision': precision, 'invalid_answers': invalid}}
            decision = confirmation_decision(results, comparisons, plan)
            self.assertFalse(decision['primary_task_gain_confirmed'])
            self.assertFalse(decision['direction_specific_gain_confirmed'])

    def test_scoring_preserves_invalid_fn_and_outside_fp_and_rejects_cache_drift(self):
        suite = ctb.load_suite()
        doc = next(d for d in suite if ctb.causal_pairs(suite[d]))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'directions.npz').write_bytes(b'frozen')
            direction_sha = sha(root / 'directions.npz')
            write_json(root / 'directions.json', {'manifest_id': 'parent', 'npz_sha256': direction_sha})
            plan = frozen_plan()
            plan['bootstrap']['draws'] = 200
            manifest = {'plan': plan, 'conditions': plan['conditions'], 'documents': [doc],
                        'source_manifest_id': 'parent', 'directions_sha256': direction_sha,
                        'source_sha256': {'suite': sha(ctb.DATA / 'ctb-suite.json')}}
            manifest['id'] = digest(manifest)
            write_json(root / 'manifest.json', manifest)
            rows = []
            for c in plan['conditions']:
                rows.append({'condition': c['name'], 'doc_id': doc, 'pass': 0,
                             'alpha': c['alpha'], 'budget': c['budget'], 'temperature': c['temperature'],
                             'manifest_id': manifest['id'], 'directions_sha256': direction_sha,
                             'condition_sha256': digest(c), 'text': 'unfinished', 'finish_reason': 'length',
                             'num_tokens': 8192, 'thinking_tokens': 8192, 'budget_forced': True})
            path = root / 'confirmation.jsonl'
            def save():
                path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            save()
            args = SimpleNamespace(out=tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                score(args)
            report = json.loads((root / 'confirmation_scores.json').read_text())
            self.assertTrue(report['complete'])
            self.assertGreater(report['conditions']['cc_0.10']['fn'], 0)
            self.assertEqual(report['conditions']['cc_0.10']['invalid_answers'], 1)
            self.assertFalse(report['decision']['primary_task_gain_confirmed'])
            rows[1].update(text='</think>[{"pair":"madeup,ids","label":"causes"}]', finish_reason='stop')
            save()
            with contextlib.redirect_stdout(io.StringIO()):
                score(args)
            result = json.loads((root / 'confirmation_scores.json').read_text())['conditions']['cc_0.10']
            self.assertEqual((result['fp'], result['outside_candidate_pairs']), (1, 1))
            rows[1]['alpha'] = .05
            save()
            with self.assertRaisesRegex(ValueError, 'generation settings'):
                load_records(root, manifest)
            rows[1]['alpha'] = .1
            rows[1]['condition_sha256'] = 'wrong'
            save()
            with self.assertRaisesRegex(ValueError, 'provenance'):
                load_records(root, manifest)
            rows[1]['condition_sha256'] = digest(plan['conditions'][1])
            rows.pop()
            save()
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                load_records(root, manifest)
            self.assertEqual(len(load_records(root, manifest, partial=True)[0]), 5)
            rows.append(copy.deepcopy(rows[0]))
            save()
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                load_records(root, manifest, partial=True)


if __name__ == '__main__':
    unittest.main()
