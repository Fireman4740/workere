"""Focused metric and raw-answer contract tests; no scientific data read."""
import itertools
import unittest
from pipelines.jev.n10 import SUBSETS, SYSTEMS, average_subset_f1, f1, parse_content, predict, evaluate


class N10Tests(unittest.TestCase):
    def test_exhaustive_subsets_and_inclusion_balance(self):
        self.assertEqual(SUBSETS, tuple(itertools.combinations(range(6), 3)))
        self.assertEqual(len(SUBSETS), 20)
        self.assertEqual([sum(i in s for s in SUBSETS) for i in range(6)], [10]*6)
        self.assertEqual(len(set(SYSTEMS)), 21)

    def test_mean_f1_is_not_f1_of_mean_counts(self):
        counts = [[1, 0, 0], [0, 0, 9]]
        self.assertEqual(average_subset_f1(counts), 50)
        self.assertAlmostEqual(f1([.5, 0, 4.5]), 100/5.5)
        self.assertNotEqual(average_subset_f1(counts), f1([.5, 0, 4.5]))

    def test_one_vote_and_legacy_last_label_wins(self):
        text = '[{"pair":"1,2","label":"causes"},{"pair":"1,2","label":"causes"},{"pair":"2,1","label":"causedby"}]'
        positive, duplicates = parse_content(text, {('1','2'),('2','1')})
        self.assertEqual(positive, {('1','2'),('2','1')})
        self.assertEqual(duplicates, 1)
        positive, _ = parse_content('[{"pair":"1,2","label":"causes"},{"pair":"1,2","label":"norel"}]', {('1','2')})
        self.assertEqual(positive, set())

    def test_invalid_answer_is_failure_not_negative(self):
        with self.assertRaises(AssertionError):
            parse_content('[{"pair":"999,2","label":"causes"}]', {('1','2')})
        with self.assertRaises(AssertionError):
            parse_content('[{"pair":"1,2","label":"maybe"}]', {('1','2')})

    def test_primary_mean_over_all_subsets_and_shared_topic_draws(self):
        # Each topic has opposing subset difficulty. Aggregated counts alone
        # cannot recover the mean of the subset-level nonlinear F1 statistic.
        per = {}
        for arm in ['ordinary','contrastive']:
            per[arm] = {'full6': {'1':[2,0,0], '2':[2,0,0]}}
            for i,name in enumerate(SYSTEMS[1:]):
                per[arm][name] = ({'1':[1,0,0], '2':[0,0,9]} if i < 10 else {'1':[0,0,9], '2':[0,0,9]})
        protocol = {'bootstrap': {'seed':17, 'draws':7, 'lower_sorted_index':0, 'upper_sorted_index':6},
                    'primary_contrasts':['ordinary','contrastive','between_arms']}
        result, _ = evaluate(per, protocol)
        expected_mean = (10*f1([1,0,9])+10*f1([0,0,18]))/20
        self.assertEqual(result['primary_contrasts'][0]['estimate_points'], 100-expected_mean)
        self.assertEqual(result['primary_contrasts'][1]['estimate_points'], 100-expected_mean)
        self.assertEqual(result['primary_contrasts'][2]['estimate_points'], 0)
        self.assertEqual(result['primary_contrasts'][2]['Bonferroni_CI98_333333_points'], [0,0])
        self.assertNotEqual(expected_mean, result['systems']['ordinary']['secondary_F1_of_mean_three_counts_points'])

    def test_six_read_fraction_and_symmetric_fusion(self):
        public = {'d': {('1','2'),('2','1')}}
        d1 = {('d','1','2'):.7, ('d','2','1'):.7}
        votes = {(a,'d',s,i): ({('1','2'),('2','1')} if s == 0 else set()) for a in ['ordinary','contrastive'] for s in [0,1] for i in range(3)}
        predictions = predict(public, d1, votes)
        # Full6 L=.5 -> fused .6, first3 L=1 -> .85, last3 L=0 -> .35.
        self.assertEqual(len(predictions['ordinary']['full6']['d']), 2)
        self.assertEqual(len(predictions['ordinary']['three_012']['d']), 2)
        self.assertEqual(predictions['ordinary']['three_345']['d'], [])


if __name__ == '__main__':
    unittest.main()
