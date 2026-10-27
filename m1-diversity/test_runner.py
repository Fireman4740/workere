"""Synthetic controller tests: no network, real labels or paid submissions."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
spec=importlib.util.spec_from_file_location('m1run',HERE/'run.py')
run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run)
from prepare import SETTINGS,requests
from preflight import toy,response

class RunnerTests(unittest.TestCase):
    def test_unknown_submission_never_repeated(self):
        rows=[dict(custom_id='toy',body={})]
        with tempfile.TemporaryDirectory() as t:
            with patch.object(run,'http',side_effect=TimeoutError('synthetic')) as h:
                with self.assertRaises(TimeoutError):run.paid_batch(rows,'shared',Path(t),{'settings':SETTINGS})
                self.assertEqual(h.call_count,1)
            with patch.object(run,'http') as h:
                with self.assertRaisesRegex(RuntimeError,'never duplicate'):run.paid_batch(rows,'shared',Path(t),{'settings':SETTINGS})
                self.assertEqual(h.call_count,0)
    def test_completed_receipt_reused_without_network(self):
        batch=dict(id='toy-batch',status='completed',results=[dict(custom_id='toy',**response())],usage={'cost':.001})
        rows=[dict(custom_id='toy',body={})]
        with tempfile.TemporaryDirectory() as t:
            with patch.object(run,'http',return_value=batch) as h:
                raw,cost=run.paid_batch(rows,'shared',Path(t),{'settings':SETTINGS})
                self.assertEqual(h.call_count,1)
            with patch.object(run,'http') as h:
                raw2,cost2=run.paid_batch(rows,'shared',Path(t),{'settings':SETTINGS})
                self.assertEqual(h.call_count,0);self.assertEqual(raw2,raw)
                self.assertEqual(cost2['paid_post_attempts'],1)
    def test_shared_once_and_absence_distinct_from_empty(self):
        d=toy(3);d['topic']='test'
        demos=[dict(doc_id=f'demo{i}',topic='other',text='<1 event>',mentions=[dict(id='1')],answer=[]) for i in range(7)]
        template=dict(system='extract',user_template='{doc_text}')
        rr,_=requests(d,demos,template,SETTINGS)
        ref={d['doc_id']:[dict(pair=p,baseline_positive=False,luna_zero=True) for p in d['pairs']]}
        one=rr[0]
        outputs=run.reconstruct(rr,{one['custom_id']:response()}, {d['doc_id']:d},SETTINGS,ref)
        self.assertEqual(outputs['integrity']['states']['VALID_EMPTY'],1)
        self.assertEqual(outputs['integrity']['states']['DELIVERY_UNKNOWN'],21)
        self.assertFalse(outputs['integrity']['all_planned_responses_valid'])
        self.assertEqual(len(outputs['integrity']['shared_custom_ids']),7)
        for row in outputs['ledgers']:
            self.assertEqual(row['O_shared']['E_valid'],1)
            self.assertEqual(row['O_shared']['V'],0);self.assertEqual(row['O_shared']['R'],0)
            self.assertEqual(row['MP']['E_valid'],0)
    def test_cost_no_double_reasoning_or_common_payment(self):
        events=[dict(channel=c,usage=dict(prompt_tokens=100,completion_tokens=50,
            completion_tokens_details={'reasoning_tokens':40},prompt_tokens_details={'cached_tokens':0})) for c in ('O_shared','O_extra','MP')]
        costs=run.resources(events,[dict(provider_cost_usd=.01,paid_post_attempts=1,get_attempts=0)],SETTINGS)
        self.assertAlmostEqual(costs['by_channel']['MP']['weighted_token_index_usd'],.000035)
        self.assertEqual(costs['total_unique_logical_requests'],3)
        self.assertEqual(costs['by_system_counterfactual']['H']['logical_requests'],2)
        self.assertTrue(costs['cost_match_total_within5pct'])

if __name__=='__main__':unittest.main()
