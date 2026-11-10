"""Scientific validity checks: edit equivalence, placebo norm, leakage, denominators."""
from __future__ import annotations

import json
import io
import hashlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from steering import ctb
from steering.weight_space import (RankOneWriter, condition_plan, counts, final_pairs,
                                  make_split, metrics, orthogonal_control,
                                  paired_interval, right_factor, unit)
from steering.wsp_pilot import score, unique_rows


class WeightSpaceTests(unittest.TestCase):
    def test_hook_equals_explicit_rank_one_weight_edit_on_every_position(self):
        torch.manual_seed(123)
        linear = torch.nn.Linear(7, 5, bias=True)
        linear.input_size, linear.output_size = 7, 5
        hook = RankOneWriter(linear, device='cpu')
        d = unit(np.arange(1., 6.))
        u = np.arange(1., 8.)
        right = right_factor(u, 2.7)
        hook.set(d, right, .1)
        x = torch.randn(2, 4, 7)
        delta = .1 * torch.outer(torch.tensor(d), torch.tensor(right))
        expected = torch.nn.functional.linear(x, linear.weight.detach() + delta, linear.bias)
        torch.testing.assert_close(linear(x), expected, atol=1e-6, rtol=1e-6)
        hook.clear()
        torch.testing.assert_close(linear(x), torch.nn.functional.linear(x, linear.weight, linear.bias),
                                   atol=0, rtol=0)

    def test_mean_input_receives_exact_requested_write(self):
        u = np.arange(1., 8.)
        right = right_factor(u, 12.)
        self.assertAlmostEqual(float(u @ right), 12., places=5)

    def test_placebo_has_same_frobenius_norm_and_is_orthogonal(self):
        d = unit(np.arange(1., 18.))
        right = right_factor(np.arange(1., 8.), 3.)
        r = orthogonal_control(d, 43)
        self.assertLess(abs(float(d @ r)), 1e-6)
        self.assertAlmostEqual(np.linalg.norm(np.outer(d, right)),
                               np.linalg.norm(np.outer(r, right)), places=6)
        self.assertFalse(np.allclose(r, orthogonal_control(d, 44)))

    def test_rejects_degenerate_statistics(self):
        for v in (np.zeros(3), np.array([1., float('nan')])):
            with self.assertRaises(ValueError): unit(v)
        for u, n in ((np.zeros(3), 1.), (np.ones(3), 0.)):
            with self.assertRaises(ValueError): right_factor(u, n)

    def test_split_has_no_eval_or_demonstration_leakage(self):
        suite, prompt = ctb.load_suite(), ctb.load_prompt()
        split = make_split(suite, prompt, ctb.eval_ids())
        self.assertEqual([len(split[k]) for k in ('fit', 'screen', 'confirmation')], [93, 20, 20])
        self.assertEqual(split, make_split(dict(reversed(list(suite.items()))), prompt, ctb.eval_ids()))
        for key in ('screen', 'confirmation'):
            self.assertFalse(set(split[key]) & set(split['demonstrations']))
        all_train = set(split['fit'] + split['screen'] + split['confirmation'])
        self.assertEqual(len(all_train), 133)
        self.assertFalse(all_train & set(ctb.eval_ids()))

    def test_final_empty_answer_does_not_resurrect_a_thinking_array(self):
        row = {'text': 'Maybe [{"pair":"ei1,ei2","label":"causes"}]\n</think>\n[]',
               'finish_reason': 'stop'}
        self.assertEqual(final_pairs(row), set())
        row['finish_reason'] = 'length'
        self.assertIsNone(final_pairs(row))

    def test_invalid_json_items_are_not_silently_valid_empty_answers(self):
        for text in ('{}', '[null]', '[{"pair":"bad","label":"causes"}]',
                     '[{"pair":"a,b","label":"unknown"}]'):
            self.assertIsNone(final_pairs({'text': '</think>' + text, 'finish_reason': 'stop'}))

    def test_ordered_binary_metric_keeps_both_orientations(self):
        row = {'text': '</think>[{"pair":"a,b","label":"causes"},'
                       '{"pair":"b,a","label":"causedby"}]', 'finish_reason': 'stop'}
        self.assertEqual(counts(final_pairs(row), {('a', 'b'), ('b', 'a')}), {'tp': 2, 'fp': 0, 'fn': 0})

    def test_two_independent_controls_at_primary_dose(self):
        conditions = condition_plan()
        self.assertEqual(len(conditions), 9)
        self.assertEqual(len({c['name'] for c in conditions}), 9)
        for family in ('cc', 'recall'):
            controls = [c for c in conditions if c['family'] == family and 'random_seed' in c]
            self.assertEqual([c['random_seed'] for c in controls], [0, 1])
        self.assertTrue(all(c['budget'] == 8192 for c in conditions))

    def test_paired_interval_rejects_missing_documents(self):
        a = {'one': {'tp': 1, 'fp': 0, 'fn': 1}}
        self.assertEqual(paired_interval(a, a, draws=100), [0., 0., 0.])
        with self.assertRaises(ValueError): paired_interval(a, {})

    def test_duplicate_rows_raise_instead_of_overwriting(self):
        with self.assertRaises(ValueError): unique_rows([{'id': 1}, {'id': 1}], lambda r: r['id'])

    def test_natural_and_forced_thinking_token_accounting(self):
        import run_vllm
        class Tokenizer:
            def __call__(self, text, **kwargs):
                return {'input_ids': [99] if text == '</think>' else [1, 2]}
            def apply_chat_template(self, messages, **kwargs):
                return 'PROMPT'
        class Engine:
            def __init__(self, completions):
                self.completions, self.pending = list(completions), []
            def add_request(self, request_id, text, params):
                self.pending.append(request_id)
            def has_unfinished_requests(self):
                return bool(self.pending)
            def step(self):
                return [SimpleNamespace(request_id=self.pending.pop(0), finished=True,
                                         outputs=[self.completions.pop(0)])]
        natural = SimpleNamespace(text='reason</think>[]', token_ids=[10, 11, 99, 12], finish_reason='stop')
        first = SimpleNamespace(text='reason', token_ids=[10, 10, 10, 10], finish_reason='length')
        answer = SimpleNamespace(text='[]', token_ids=[12], finish_reason='stop')
        for completions, expected in (([natural], (2, 1, 4)), ([first, answer], (4, 1, 5))):
            sink = io.StringIO()
            engine = Engine(completions)
            with patch.object(run_vllm, 'request_params', return_value=None):
                run_vllm.generate_condition(SimpleNamespace(llm_engine=engine), Tokenizer(),
                    [{'doc_id': 'd', 'pass': 0, 'messages': []}],
                    {'name': 'test', 'budget': 4, 'answer_tokens': 2}, 6, sink)
            row = json.loads(sink.getvalue())
            self.assertEqual(tuple(row[k] for k in ('thinking_tokens', 'answer_generated_tokens', 'num_tokens')), expected)

    def test_scorer_keeps_failed_documents_in_denominator_and_refuses_missing_rows(self):
        suite = ctb.load_suite()
        doc = next(d for d in suite if ctb.causal_pairs(suite[d]))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {'id': 'test', 'split': {'screen': [doc]}, 'conditions': condition_plan()}
            (root / 'manifest.json').write_text(json.dumps(manifest))
            (root / 'directions.npz').write_bytes(b'fixture')
            directions_sha = hashlib.sha256(b'fixture').hexdigest()
            (root / 'directions.json').write_text(json.dumps({'manifest_id': 'test', 'npz_sha256': directions_sha}))
            row = {'condition': 'base', 'doc_id': doc, 'text': 'unfinished', 'finish_reason': 'length',
                   'num_tokens': 8192, 'budget_forced': True, 'manifest_id': 'test', 'directions_sha256': directions_sha}
            (root / 'screen.jsonl').write_text(json.dumps(row) + '\n')
            args = SimpleNamespace(out=tmp)
            score(args, partial=True)
            result = json.loads((root / 'screen_scores.json').read_text())['conditions']['base']
            self.assertEqual(result['documents'], 1)
            self.assertEqual(result['invalid_answers'], 1)
            self.assertGreater(result['fn'], 0)
            self.assertEqual(result['f1'], 0.)
            with self.assertRaises(ValueError): score(args)
            row.update(finish_reason='stop', text='</think>[{"pair":"invented,ids","label":"causes"}]')
            (root / 'screen.jsonl').write_text(json.dumps(row) + '\n')
            score(args, partial=True)
            result = json.loads((root / 'screen_scores.json').read_text())['conditions']['base']
            self.assertEqual(result['fp'], 1)
            self.assertEqual(result['outside_candidate_pairs'], 1)
            row['directions_sha256'] = 'stale'
            (root / 'screen.jsonl').write_text(json.dumps(row) + '\n')
            with self.assertRaises(ValueError): score(args, partial=True)


if __name__ == '__main__':
    unittest.main()
