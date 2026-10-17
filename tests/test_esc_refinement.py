import json
import tempfile
import unittest
from pathlib import Path
from pipelines.jev.refinement import _load, aggregate_chains, lodo_thresholds, symmetrize
from pipelines.jev.coreference import collect, connected_components, unordered_pairs


class EscRefinement(unittest.TestCase):
    def test_dedicated_requests_exclude_causal_question_and_resume_from_cache(self):
        from benchmark.protocols import public_inputs
        from benchmark.suite import freeze
        from tests.test_benchmark import config, rows
        suite = freeze(config(),rows())
        n = sum(len(unordered_pairs(i.reference_entities,i.reference_candidates))
                for i in public_inputs(suite,'relations_gold'))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg = {'api_calls':n,'max_workers':3,'coreference_client':{
                'model':'liquid/d1-ultra','model_version':'d1-ultra-20261015',
                'mode':'mock','cache_path':str(root/'cache.sqlite'),'max_calls':100}}
            first, second = root/'first',root/'second'
            first.mkdir();second.mkdir()
            clusters,usage = collect(cfg,suite,first)
            replay,replay_usage = collect(cfg,suite,second)
            self.assertEqual(clusters,replay)
            self.assertEqual(usage['logical_queries'],n)
            self.assertEqual(replay_usage['cache_hits'],n)
            for line in (first/'coreference-decisions.jsonl').read_text().splitlines():
                request = json.loads(line)['request']
                self.assertEqual(set(request['questions']),{'same_event'})
                self.assertNotIn('"gold"',request['state'])

    def test_predicted_chains_close_transitively_with_fixed_threshold_and_singletons(self):
        clusters = connected_components(['a','b','c','d'], [('a','b',.5),('b','c',.7),('a','c',.1),('c','d',.49)])
        self.assertEqual(clusters['a'],clusters['c'])
        self.assertNotEqual(clusters['a'],clusters['d'])
    def test_effective_frozen_rules_can_be_written_to_json_receipt(self):
        config = Path(__file__).resolve().parents[1] / 'experiments/jev/active.yaml'
        rules = _load(config)[2]
        restored = json.loads(json.dumps(rules, allow_nan=False))
        self.assertEqual(restored['registered'], '2026-10-16')
        self.assertEqual(restored['threshold']['grid'], [i / 20 for i in range(1,20)])

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
