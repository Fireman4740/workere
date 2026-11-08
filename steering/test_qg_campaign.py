"""Scientific invariants of QG; no real API calls."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from steering import qg_protocol as q
from steering import qg_campaign as run
from steering import ap_campaign as ap
from steering import ap_protocol as p
from steering.test_ap_campaign import fixture, FakeEngine
from steering.test_campaign_api import Tokenizer


def panel():
    selected, docs, rules, base = fixture()
    for values in base.values():
        for pairs in values.values():
            pairs.add(('a', 'b'))
    historical = copy.deepcopy(base)
    views = {(ds, d, o): {('a', 'b'): True, ('a', 'c'): o == 0}
             for ds, ids in selected.items() for d in ids for o in (0, 1)}
    systems = q.candidate_systems(selected, base, historical, views)
    return selected, docs, rules, base, historical, systems, q.blocks(selected, docs, systems)


def row_for(block, order, method, vector):
    shown = q.presentation(block, order)
    if method == 'select':
        index = block['options'].index(list(vector))
        parsed = {'choice': f"G{shown['option_order'].index(index)}"}
    else:
        parsed = {str(i): 'Y' if vector[j] else 'N' for i, j in enumerate(shown['pair_order'])}
    return dict(status='valid', parsed=parsed)


class QGTests(unittest.TestCase):
    def test_gold_and_coreference_do_not_affect_requests(self):
        selected, docs, rules, base, historical, systems, blocks = panel()
        changed = copy.deepcopy(docs)
        for values in changed.values():
            for doc in values.values():
                doc['gold'] = ['DO_NOT_USE_THIS_LABEL']
                doc['chains'] = [['DO_NOT_USE_COREFERENCE']]
        second = q.blocks(selected, changed, systems)
        self.assertEqual(blocks, second)
        for method in ('select', 'edit'):
            jobs = run.jobs_for(blocks, rules, method)
            self.assertEqual(jobs, run.jobs_for(second, rules, method))
            self.assertNotIn('DO_NOT_USE', str(jobs))

    def test_options_are_deduplicated_and_source_names_absent(self):
        _, _, rules, _, _, _, blocks = panel()
        for b in blocks:
            self.assertEqual(len(b['options']), len(set(map(tuple, b['options']))))
            self.assertIn([False]*len(b['pairs']), b['options'])
            text = str(q.messages(b, 0, 'select', rules[b['dataset']]))
            self.assertNotIn('screen0', text)
            self.assertNotIn('retained', text)
            self.assertNotIn('historical', text)

    def test_screens_invalid_is_fallback_not_negative(self):
        selected, docs, rules, base = fixture()
        base['CTB']['c0'].add(('a', 'b'))
        systems = q.candidate_systems(selected, base, copy.deepcopy(base),
                                     {('CTB', 'c0', 0): {('a', 'b'): None}})
        self.assertEqual(systems['screen0']['CTB']['c0'], {('a', 'b')})
        self.assertEqual(systems['screen1']['CTB']['c0'], {('a', 'b')})

    def test_block_partition_covers_union_once(self):
        selected, docs, rules, base, historical, systems, blocks = panel()
        for ds, ids in selected.items():
            for d in ids:
                pairs = [tuple(pair) for b in blocks if b['dataset'] == ds and b['doc_id'] == d for pair in b['pairs']]
                expected = set().union(*(s[ds][d] for s in systems.values()))
                self.assertEqual(set(pairs), expected)
                self.assertEqual(len(pairs), len(set(pairs)))
        self.assertTrue(all(len(b['pairs']) <= 8 for b in blocks))

    def test_permutation_decodes_back_to_canonical_pairs(self):
        *_, blocks = panel()
        for b in blocks:
            for vector in b['options']:
                for method in ('select', 'edit'):
                    for order in (0, 1):
                        self.assertEqual(q.decode(b, order, method, row_for(b, order, method, vector)), tuple(vector))
            self.assertEqual(q.presentation(b, 1)['option_order'], q.presentation(b, 0)['option_order'][::-1])

    def test_invalid_or_disagreement_preserves_reference(self):
        _, _, _, base, _, _, blocks = panel()
        for method in ('select', 'edit'):
            rows = {}
            for b in blocks:
                rows[(b['key'], 0)] = row_for(b, 0, method, b['options'][-1])
                rows[(b['key'], 1)] = dict(status='invalid_answer', parsed=None)
            out, _, quality = q.combine(base, blocks, rows, method)
            self.assertEqual(out, base)
            self.assertGreater(quality['ESC']['invalid_fallback_pairs'], 0)
            # Complete but contrary vectors also abstain on every disagreement.
            for b in blocks:
                rows[(b['key'], 1)] = row_for(b, 1, method, b['options'][0])
            out, _, quality = q.combine(base, blocks, rows, method)
            self.assertEqual(out, base)

    def test_full_denominator_and_same_budgets(self):
        selected, docs, rules, base, historical, systems, blocks = panel()
        for method in ('select', 'edit'):
            jobs = run.jobs_for(blocks, rules, method)
            self.assertEqual(len(jobs), 2*len(blocks))
            self.assertTrue(all((j['thinking'], j['answer_tokens']) == (2048, 256) for j in jobs))
            outputs = {j['key']: dict(status='invalid_answer', parsed=None) for j in jobs}
            predictions, report = run.evaluate(blocks, jobs, outputs, method, base, historical, systems, docs)
            self.assertEqual(predictions, base)
            self.assertEqual(report['metrics']['CTB']['documents'], len(selected['CTB']))
            self.assertEqual(report['metrics']['ESC']['delta'], (0.0, 0.0, 0.0))

    def test_exact_shape_and_output_validation(self):
        *_, blocks = panel()
        b = blocks[0]
        self.assertIsNone(q.decode(b, 0, 'select', dict(status='valid', parsed={'choice':'G999'})))
        self.assertIsNone(q.decode(b, 0, 'edit', dict(status='valid', parsed={'0':'Y'})))

    def test_repeat_uses_fresh_namespace(self):
        _, _, rules, base, _, _, blocks = panel()
        jobs = run.jobs_for(blocks, rules, 'select')
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            fake = FakeEngine()
            outputs = fake.execute(jobs, out/'graphs')
            with patch.object(ap, 'Engine', return_value=fake) as engine:
                result = run.repeat_audit(out, out, 9999999999, 'select', blocks, jobs, outputs, base)
            self.assertTrue(engine.call_args.kwargs['namespace'].endswith('-repeat'))
            self.assertEqual(result['ESC']['changed_binary_decisions'], 0)

    def test_persistent_clock_not_reset_by_retry(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(run.time, 'time', return_value=1000):
            first = run.deadline(Path(tmp), 'select')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(run.time, 'time', return_value=1000): initial = run.deadline(root, 'select')
            with patch.object(run.time, 'time', return_value=8000): retry = run.deadline(root, 'select')
            self.assertEqual(initial, retry)
            self.assertEqual(first, initial)

    def test_driver_roundtrip_including_cross_arm_and_repeat(self):
        data = panel()
        class Engine(FakeEngine):
            tokenizer = Tokenizer()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'qg'
            root.mkdir()
            ap.save(root/'manifest.json', {'synthetic': True})
            ap.save(root/'blocks.json', data[-1])
            with patch.object(run, 'prepare', return_value=data), patch.object(ap, 'Engine', Engine):
                for method in ('select', 'edit'):
                    run.run(root, Path(tmp)/'old', Path(tmp)/'c2', Path(tmp), method)
                    self.assertEqual(ap.read(root/method/'terminal.json')['status'], 'complete')
            report = ap.read(root/'edit/report.json')
            self.assertEqual(report['paired_vs_other_arm']['other'], 'select')
            self.assertEqual(report['metrics']['ESC']['documents'], len(data[0]['ESC']))
            self.assertEqual(report['repeatability']['ESC']['changed_binary_decisions'], 0)


if __name__ == '__main__':
    unittest.main()
