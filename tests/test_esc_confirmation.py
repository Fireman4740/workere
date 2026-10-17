import copy
import json
import tempfile
import unittest
from pathlib import Path
from benchmark.io import fingerprint
from benchmark.suite import freeze
from pipelines.jev.confirmation import collect, predictions, project_scores, paired_bootstrap, _load
from pipelines.jev.state_builder import noul_question
from tests.test_benchmark import config, rows
from benchmark.protocols import public_inputs, score_predictions

def fixture():
    suite=freeze(config(),rows())
    for doc in suite["documents"]:
        doc["pair_list"]=[list(p) for p in dict.fromkeys(tuple(pair) for a,b in doc["pair_list"] for pair in ((a,b),(b,a)))]
    suite.pop("fingerprint");suite["fingerprint"]=fingerprint(suite)
    return suite

class FrozenConfirmation(unittest.TestCase):
    def test_inference_predictions_and_threshold_are_independent_of_confirmation_labels(self):
        suite=fixture()
        responses=[]
        for item in public_inputs(suite,'relations_gold'):
            responses.extend({'id':item.document.id,'source':a,'target':b,'positive_probability':.7}
                             for a,b in item.reference_candidates)
        before=predictions(suite,responses,symmetric=True,threshold=.65)
        other=copy.deepcopy(suite)
        for doc in other['documents']:doc['gold']=[]
        other.pop('fingerprint');other['fingerprint']=fingerprint(other)
        self.assertEqual(before,predictions(other,responses,symmetric=True,threshold=.65))
        self.assertTrue(all(row['metadata']['threshold']==.65 for row in before))

    def test_ordered_baseline_and_mean_use_distinct_fixed_rules(self):
        raw={('a','b'):.9,('b','a'):.3}
        self.assertEqual(project_scores(raw,symmetric=False),raw)
        mean=project_scores(raw,symmetric=True)
        self.assertAlmostEqual(mean['a','b'],.6)
        self.assertEqual(mean['a','b'],mean['b','a'])
        with self.assertRaises(ValueError):project_scores({('a','b'):.9},symmetric=True)

    def test_mock_collector_replays_without_additional_calls_and_official_scorer_accepts(self):
        suite=fixture();frozen={'context_style':'document','causal_questions':noul_question()}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            cfg={'max_workers':4,'client':{'model':'test/mock','model_version':'v1','mode':'mock',
                 'cache_path':str(root/'cache.sqlite'),'requests_per_minute':100000,'max_calls':100}}
            first=root/'first';first.mkdir();second=root/'second';second.mkdir()
            r1,s1=collect(cfg,frozen,suite,first);r2,s2=collect(cfg,frozen,suite,second)
            self.assertEqual(r1,r2)
            self.assertEqual(s1['logical_queries'],len(r1))
            self.assertEqual(s2['api_calls'],0)
            self.assertEqual(s2['cache_hits'],len(r1))
            a=score_predictions(suite,predictions(suite,r1,symmetric=False,threshold=.5),'relations_gold')
            b=score_predictions(suite,predictions(suite,r1,symmetric=True,threshold=.65),'relations_gold')
            result=paired_bootstrap(a,b,{'samples':20,'seed':42,'confidence_level':.95})
            self.assertFalse(result['threshold_refitted'])

    def test_sealed_input_mutation_is_rejected_before_inference(self):
        # Hash validation precedes parsing suite data and construction of an API client.
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'freeze.json').write_text('{}')
            (root/'cfg.yaml').write_text('schema: workere-esc-confirmation/v1\nfreeze: freeze.json\nfreeze_sha256: wrong\n')
            with self.assertRaisesRegex(ValueError,'freeze hash mismatch'):_load(root/'cfg.yaml')

if __name__=='__main__':unittest.main()
