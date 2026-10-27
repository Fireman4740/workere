import copy
import itertools
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eci_m0_pilot.m0 import as_pairs, build, permutations, validate
from eci_m0_pilot.protocol import ARMS, decode_response, evaluate, ledger_for, remap_tags, remap_template, requests_for
from eci_m0_pilot.run_pilot import batch_run

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / 'config.json').read_text())

def doc(n=8):
    ids = list(map(str, range(1, n+1)))
    return {'doc_id': 'toy', 'topic': 'test', 'mentions': [{'id': x, 'sentence': int(x) % 3} for x in ids],
            'pairs': list(itertools.combinations(ids, 2)),
            'text': '12 people in 2017. ' + ' '.join(f'<{x} event>' for x in ids)}

def response(text='[]', **kwargs):
    body = {'model': CFG['allowed_served_models'][0], 'choices': [{'finish_reason': 'stop', 'message': {'content': text}}]}
    body.update(kwargs)
    return {'response': {'status_code': 200, 'body': body}}

class ProtocolTests(unittest.TestCase):
    def test_random_partitions_sparse_dense_and_unknown(self):
        rng = random.Random(42)
        for _ in range(150):
            d = doc(rng.randrange(0, 30)); k = rng.randrange(1, 12)
            d['pairs'] = [p for p in d['pairs'] if rng.random() < .7]
            for m in d['mentions']:
                if rng.random() < .1: m['sentence'] = None
            rows = build(d, k, 42); validate(rows, d, k)
            self.assertEqual(rows, build(d, k, 42))
            self.assertTrue(all(r['status'] == 'NOT_SENT' for r in rows))
    def test_zero_gold_invariance(self):
        d = doc(); a = build(d)
        d.update(gold_pairs=[['1', '2']], baseline_positive=[['2', '3']])
        self.assertEqual(a, build(d))
    def test_source_data_do_not_feed_gold(self):
        public = [json.loads(s) for s in (HERE/'data/public_train233.jsonl').read_text().splitlines()]
        for d in public: self.assertEqual(set(d), {'doc_id','topic','mentions','pairs','text'})
    def test_invalid_mentions_rejected(self):
        d = doc(); d['mentions'].append(d['mentions'][0])
        with self.assertRaises(ValueError): build(d)
        d = doc(); d['pairs'].append(('100','1'))
        with self.assertRaises(ValueError): build(d)
    def test_remapping_never_replaces_numbers_in_document(self):
        t = '12 people in 2017 with <1 explosion> and <2 injury>.'
        pi = {'1':'2','2':'1'}
        self.assertEqual(remap_tags(t,pi), '12 people in 2017 with <2 explosion> and <1 injury>.')
        self.assertEqual(remap_tags(remap_tags(t,pi),pi),t)
        self.assertEqual(remap_template('<1 x> → ID is "1"; "1,2"; 12 people', pi), '<2 x> → ID is "2"; "2,1"; 12 people')
    def test_real_render_all_four_conditions(self):
        docs={d['doc_id']:d for d in map(json.loads,(HERE/'data/public_train233.jsonl').read_text().splitlines())}
        selection=json.loads((HERE/'data/selection.json').read_text())
        demos=json.loads((HERE/'data/demonstrations.json').read_text()); template=json.loads((HERE/'data/prompt.json').read_text())
        for did in selection['document_ids']:
            rr=requests_for(docs[did],demos,selection['demonstration_schedule'],template,CFG)
            self.assertEqual(len(rr),28)
            self.assertTrue(all(r['demo_id'] != did for r in rr))
            for k in range(1,8): self.assertEqual(len({r['demo_id'] for r in rr if r['pass']==k}),1)
    def test_empty_and_no_exposure_are_distinct(self):
        d=doc(); rows=build(d); row=next(r for r in rows if r['arm']=='M')
        for i,r in enumerate(rows): r.update(custom_id=str(i),prompt_sha256='hash')
        led,events=ledger_for(d,rows,{},CFG)
        self.assertTrue(all(x['E_submitted']==0 for x in led['M']))
        row['custom_id']='m1';row['prompt_sha256']='hash'
        led,events=ledger_for(d,[row],{'m1':response()},CFG)
        self.assertEqual(events[0]['state'],'VALID_EMPTY')
        self.assertTrue(all(x['V']==0 for x in led['M']))
        self.assertEqual(sum(x['E_valid'] for x in led['M']),len(row['allowed_original']))
    def test_inverse_then_deduplicate(self):
        d=doc(3); row=build(d)[0];row['id_mapping']={'1':'3','2':'1','3':'2'}
        out=decode_response(row,response('[{"pair":"3,1","label":"causes"},{"pair":"1,3","label":"causedby"}]'),CFG)
        self.assertEqual(out['positive'],[('1','2')]);self.assertEqual(out['duplicates_removed'],1)
    def test_invalid_not_negative(self):
        row=build(doc())[0]
        for text in ('garbage','[7]','[{"pair":"1,999","label":"causes"}]','[{"pair":"1,2","label":"wrong"}]','[{"label":"causes"}]'):
            out=decode_response(row,response(text),CFG)
            self.assertFalse(out['valid']);self.assertTrue(out['delivered']);self.assertEqual(out['positive'],[])
    def test_final_empty_not_shadowed_by_reasoning_array(self):
        row=build(doc())[0]
        out=decode_response(row,response('An example: [{"pair":"1,2","label":"causes"}]. Final: []'),CFG)
        self.assertEqual(out['state'],'INVALID_FORMAT')
    def test_off_mask_filtered_and_explicit_reject_distinct(self):
        row=build(doc())[0];row['allowed_original']=[['1','2']]
        out=decode_response(row,response('[{"pair":"1,3","label":"causes"},{"pair":"1,2","label":"norel"}]'),CFG)
        self.assertTrue(out['valid']);self.assertEqual(out['outside_mask'],[('1','3')]);self.assertEqual(out['explicit_reject'],[('1','2')])
    def test_truncation_quarantined(self):
        r=response();r['response']['body']['choices'][0]['finish_reason']='length'
        self.assertEqual(decode_response(build(doc())[0],r,CFG)['state'],'INCOMPLETE')
    def test_exact_f1_formula_and_no_baseline_removal(self):
        d=doc(3);rows=build(d)
        for i,r in enumerate(rows):r.update(custom_id=str(i),prompt_sha256='h')
        results={str(i):response('[{"pair":"1,2","label":"causes"}]') for i in range(len(rows))}
        led,_=ledger_for(d,rows,results,CFG)
        ev=[{'doc_id':'toy','pair':['1','2'],'gold':True,'baseline_positive':False,'luna_zero':True},
            {'doc_id':'toy','pair':['1','3'],'gold':True,'baseline_positive':True,'luna_zero':False},
            {'doc_id':'toy','pair':['2','3'],'gold':False,'baseline_positive':True,'luna_zero':False}]
        scores=evaluate(ev,{'toy':led})
        self.assertEqual(scores['arms']['O']['f1'],.8)
        self.assertEqual(scores['arms']['O']['removed_true'],0)
        self.assertEqual(scores['arms']['O']['true_additions'],1)
    def test_ambiguous_submission_not_retried(self):
        with tempfile.TemporaryDirectory() as td:
            with patch('eci_m0_pilot.run_pilot.http',side_effect=TimeoutError('synthetic timeout')) as http:
                with self.assertRaises(TimeoutError):batch_run([{'custom_id':'x','body':{}}],CFG,Path(td))
                self.assertEqual(http.call_count,1)
            with patch('eci_m0_pilot.run_pilot.http') as http:
                with self.assertRaisesRegex(RuntimeError,'receipt prevents'):batch_run([{'custom_id':'x','body':{}}],CFG,Path(td))
                self.assertEqual(http.call_count,0)
    def test_paid_budget_checked_before_post(self):
        cfg={**CFG,'max_logical_requests':0}
        with tempfile.TemporaryDirectory() as td, patch('eci_m0_pilot.run_pilot.http') as http:
            with self.assertRaises(RuntimeError):batch_run([{'custom_id':'x','body':{}}],cfg,Path(td))
            self.assertEqual(http.call_count,0)

if __name__=='__main__':unittest.main()
