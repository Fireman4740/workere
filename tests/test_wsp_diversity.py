"""Vote ties, distinct readers, corpus denominator, cost, and oracle semantics."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from steering.wsp_diversity import audit_predictions, load_source, strict_vote


class DiversityTests(unittest.TestCase):
    def test_strict_majority_rejects_ties_and_duplicate_pairs_do_not_add_votes(self):
        p = ('a', 'b')
        self.assertEqual(strict_vote([[p, p, p], set(), set()]), set())
        self.assertEqual(strict_vote([{p}, {p}, set(), set()]), set())
        self.assertEqual(strict_vote([{p}, {p}, {p}, set()]), {p})
        with self.assertRaises(ValueError):
            strict_vote([])

    def fixture(self):
        plan = json.loads((Path(__file__).parents[1] / 'steering/wsp_diversity_plan.json').read_text())
        names = plan['conditions']
        g1, g2, fp = ('a', 'b'), ('c', 'd'), ('x', 'y')
        predictions = {n: {'positive': set(), 'negative': set()} for n in names}
        for n in ('base', 'cc_0.10', 'cc_random0_0.10', 'recall_random0_0.10'):
            predictions[n]['positive'] = {g1}
        predictions['cc_0.10']['positive'].add(g2)
        predictions['base']['negative'].add(fp)
        tokens = {n: {'positive': 10, 'negative': 10} for n in names}
        gold = {'positive': {g1, g2}, 'negative': set()}
        return ['positive', 'negative'], gold, predictions, tokens, plan

    def test_union_ceiling_is_not_majority_recall_and_cost_counts_every_reading(self):
        args = self.fixture()
        result = audit_predictions(*args)
        majority = result['conditions']['vote_all6']
        self.assertEqual((majority['tp'], majority['fp'], majority['fn'], majority['documents']), (1, 0, 1, 2))
        self.assertEqual(majority['recall'], .5)
        self.assertEqual(result['union_diagnostic']['recall'], 1.)
        self.assertEqual(result['union_diagnostic']['fp'], 1)
        self.assertEqual(result['union_diagnostic']['gold_positive_documents'], 1)
        self.assertEqual(majority['emitted_tokens_total'], 120)
        self.assertEqual(majority['emitted_tokens_ratio_to_single_base'], 6.)
        self.assertEqual(result['conditions']['vote_randoms4']['tp'], 0)
        self.assertEqual(result['reader_contributions']['cc_0.10']['gold_unique_to_this_reader_ordered'], 1)
        both_empty = next(p for p in result['pairwise'] if p['a'] == 'cc_random1_0.10' and p['b'] == 'recall_random1_0.10')
        self.assertIsNone(both_empty['edge_jaccard'])
        self.assertEqual(both_empty['both_empty_documents'], 2)

    def test_no_dropping_documents_repeated_readers_or_tuned_vote_threshold(self):
        ids, gold, predictions, tokens, plan = self.fixture()
        missing = copy.deepcopy(predictions)
        del missing['base']['negative']
        with self.assertRaisesRegex(ValueError, 'retain every document'):
            audit_predictions(ids, gold, missing, tokens, plan)
        for change in ('duplicate', 'threshold'):
            changed = copy.deepcopy(plan)
            if change == 'duplicate':
                changed['votes'][0]['members'].append('cc_0.10')
            else:
                changed['votes'][0]['minimum_votes'] = 2
            with self.assertRaisesRegex(ValueError, 'strict-majority'):
                audit_predictions(ids, gold, predictions, tokens, changed)

    def test_source_fingerprint_is_checked_before_using_cached_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'manifest.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'cached source changed'):
                load_source(tmp, {'file_hashes': {'manifest.json': '0' * 64}}, ['base'])


if __name__ == '__main__':
    unittest.main()
