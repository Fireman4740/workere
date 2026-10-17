import unittest
from pipelines.jev.refinement import aggregate_chains, lodo_thresholds, symmetrize


class EscRefinement(unittest.TestCase):
    def test_held_out_labels_never_change_its_fitted_threshold(self):
        values = {'a': [.2,.8], 'b': [.3,.7], 'held': [.1,.9]}
        labels = {'a': [False,True], 'b': [False,True], 'held': [False,True]}
        grid = [.25,.5,.75,.95]
        before = lodo_thresholds(values,labels,grid)['held']
        labels['held'] = [True,False]
        self.assertEqual(lodo_thresholds(values,labels,grid)['held'],before)

    def test_threshold_ties_choose_largest_registered_value(self):
        values = {'a': [.1], 'b': [.1]}
        labels = {'a': [False], 'b': [False]}
        self.assertEqual(lodo_thresholds(values,labels,[.5,.75,.95]),{'a':.95,'b':.95})

    def test_chain_max_diffuses_evidence_and_rejects_internal_links(self):
        raw = {('a','b'):.9,('b','a'):.7,('a','c'):.2,('c','a'):.4,
               ('b','c'):.8,('c','b'):.6}
        result = aggregate_chains(symmetrize(raw),{'a':0,'b':0,'c':1})
        self.assertEqual(result['a','b'],0)
        self.assertEqual(result['b','a'],0)
        for pair in [('a','c'),('c','a'),('b','c'),('c','b')]:
            self.assertAlmostEqual(result[pair],.7)


if __name__=='__main__':unittest.main()
