"""Synthetic software tests ONLY. No API calls, no ESC performance measurements."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agent4_v0 import adapter
from agent4_v0.common import digest, jsonl, save, sha, PreflightError
from agent4_v0.kit import pilot
from agent4_v0.convert_cache import convert
from agent4_v0.review import certify


def fixture():
    text='Fire made smoke. Rain delayed play. Wind broke roofs. Snow stopped cars.'
    mentions=[]
    for i,word in enumerate(('Fire','smoke','Rain','play','Wind','roofs','Snow','cars')):
        start=text.index(word)
        mentions.append({'id':str(i+1),'surface':word,'start':start,'end':start+len(word)})
    pairs=[{'a':a['id'],'b':b['id'],'baseline_score':.8}
           for a in mentions for b in mentions if a['id'] < b['id']]
    doc={'doc_id':'1_synthetic','topic_id':'1','split':'dev','text':text,'mentions':mentions,'pairs':pairs}
    doc['original_marked_text']=adapter.markup(text,mentions)
    mapping={m['id']:m['id'] for m in mentions}
    task={'task_id':'T','key':'1_synthetic::1::2','doc_id':doc['doc_id'],'variant':'original',
          'document_text':text,'text_sha256':hashlib.sha256(text.encode()).hexdigest(),
          'mentions':mentions,'id_map':mapping,'exposed_pair':['1','2'],
          'candidate_pairs':[[p['a'],p['b']] for p in pairs]}
    req={'model':'openai/gpt-6-luna:batch','reasoning':{'effort':'high'},'max_tokens':32768,
         'messages':[{'role':'system','content':'frozen'}, {'role':'user','content':'DEMO'},
                     {'role':'assistant','content':'[]'}, {'role':'user','content':'Text:\n'+doc['original_marked_text']}]}
    snap={'doc_id':doc['doc_id'],'pass_index':0,'demonstration_id':'2_synthetic_demo',
          'request':req,'request_sha256':digest(req)}
    return doc,task,snap


def response(text='[]',finish='stop',model='openai/gpt-6-luna-20260922'):
    return {'model':model,'choices':[{'finish_reason':finish,'message':{'content':text}}],
            'usage':{'prompt_tokens':10,'completion_tokens':2,'cost':0}}


class Agent4AdapterTests(unittest.TestCase):
    def test_original_request_is_byte_semantically_identical(self):
        d,t,s=fixture()
        self.assertEqual(adapter.payload(t,s,d),s['request'])

    def test_all_mentions_remapped_on_length_change(self):
        d,t,s=fixture()
        start=d['text'].index('made')
        text,mentions=pilot.apply_edits(d['text'],d['mentions'],[{'start':start,'end':start+4,'before':'made','after':'preceded'}])
        t.update(document_text=text,mentions=mentions,variant='target',text_sha256=hashlib.sha256(text.encode()).hexdigest())
        body=adapter.payload(t,s,d)
        self.assertIn('<2 smoke>',body['messages'][-1]['content'])
        self.assertEqual(body['messages'][:-1],s['request']['messages'][:-1])

    def test_rename_restores_pair_identity(self):
        d,t,s=fixture()
        m={x['id']:str(9-int(x['id'])) for x in d['mentions']}
        t.update(variant='rename_ids',id_map=m,mentions=[{**x,'id':m[x['id']]} for x in d['mentions']],
                 exposed_pair=[m['1'],m['2']],candidate_pairs=[[m[a],m[b]] for a,b in t['candidate_pairs']])
        self.assertIn('<8 Fire>',adapter.payload(t,s,d)['messages'][-1]['content'])
        r=adapter.prediction(t,response('[{"pair":"8,7","label":"causes"}]'))
        self.assertEqual((r['vote_ab'],r['vote_ba']),(1,0))

    def test_missing_candidate_rejected(self):
        d,t,s=fixture();t['candidate_pairs'].pop()
        with self.assertRaises(ValueError):adapter.payload(t,s,d)

    def test_missing_mention_rejected(self):
        d,t,s=fixture();t['mentions']=t['mentions'][:-1]
        with self.assertRaises(ValueError):adapter.payload(t,s,d)

    def test_snapshot_tampering_rejected(self):
        d,t,s=fixture();s['request']['max_tokens']=5
        with self.assertRaises(ValueError):adapter.payload(t,s,d)

    def test_same_topic_demo_rejected(self):
        d,t,s=fixture();s['demonstration_id']='1_other'
        with self.assertRaises(ValueError):adapter.payload(t,s,d)

    def test_valid_omission_is_zero(self):
        _,t,_=fixture();r=adapter.prediction(t,response())
        self.assertEqual((r['status'],r['vote_ab'],r['vote_ba']),('ok',0,0))

    def test_bad_output_is_missing_not_zero(self):
        _,t,_=fixture()
        for resp in (response('not JSON'),response('[]','length'),response('[]',model='other/model'),None):
            r=adapter.prediction(t,resp)
            self.assertNotEqual(r['status'],'ok')
            self.assertNotIn('predicted_ordered_pairs',r)

    def test_duplicate_pair_last_label_wins_as_reference(self):
        _,t,_=fixture()
        r=adapter.prediction(t,response('[{"pair":"1,2","label":"causes"},{"pair":"1,2","label":"norel"}]'))
        self.assertEqual(r['vote_ab'],0)

    def test_atomic_nonrefundable_budget_and_no_resubmit(self):
        with tempfile.TemporaryDirectory() as td:
            ledger=adapter.Ledger(Path(td)/'budget.sqlite',1,'test')
            ledger.reserve('one',.6)
            with self.assertRaises(ValueError):ledger.reserve('two',.6)
            with self.assertRaises(ValueError):ledger.reserve('one',.1)
            ledger.reserve('three',.4)
            ledger.db.close()

    def test_budget_is_persistent_across_adapter_instances(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'budget.sqlite'
            first=adapter.Ledger(path,1,'test');first.reserve('first',.7);first.db.close()
            second=adapter.Ledger(path,1,'test')
            with self.assertRaises(ValueError):second.reserve('second',.4)
            with self.assertRaises(ValueError):second.reserve('first',.1)
            second.db.close()

    def test_global_v0_call_limit_is_84(self):
        with tempfile.TemporaryDirectory() as td:
            ledger=adapter.Ledger(Path(td)/'budget.sqlite',1000,'test')
            for i in range(84):ledger.reserve(str(i),.01)
            with self.assertRaisesRegex(ValueError,'global V0 call limit'):
                ledger.reserve('85th',.01)
            ledger.db.close()

    def test_no_pricing_no_claimed_ceiling(self):
        _,_,s=fixture()
        with self.assertRaises(ValueError):adapter.reservation(s['request'],{})

    def test_dry_run_never_calls_transport(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);d,t,s=fixture()
            jsonl(p/'tasks',[]);jsonl(p/'baseline',[d]);jsonl(p/'snaps',[s])
            save(p/'selection',{'baseline_sha256':sha(p/'baseline'),'selected':[]})
            a=SimpleNamespace(out=str(p/'out'),tasks=p/'tasks',baseline=p/'baseline',snapshots=p/'snaps',
                              selection=p/'selection',review_certificate=None,budget=None,execute=False)
            with patch('agent4_v0.adapter.live',side_effect=AssertionError('network called')):
                adapter.run(a)
            pf=json.loads((p/'out/preflight.json').read_text())
            self.assertTrue(any('BLOCKED_BUDGET' in x for x in pf['blockers']))
            self.assertFalse((p/'out/predictions.jsonl').exists())

    def test_execute_without_review_or_budget_never_calls_transport(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);d,t,s=fixture()
            jsonl(p/'tasks',[]);jsonl(p/'baseline',[d]);jsonl(p/'snaps',[s])
            save(p/'selection',{'baseline_sha256':sha(p/'baseline'),'selected':[]})
            a=SimpleNamespace(out=str(p/'out'),tasks=p/'tasks',baseline=p/'baseline',snapshots=p/'snaps',
                              selection=p/'selection',review_certificate=None,budget=None,execute=True)
            with patch('agent4_v0.adapter.live',side_effect=AssertionError('network called')):
                with self.assertRaisesRegex(ValueError,'PRECONDITIONS_BLOCKED'):
                    adapter.run(a)
            self.assertFalse((p/'out/predictions.jsonl').exists())

    def test_no_empty_pseudo_score(self):
        from agent4_v0.score import score
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);jsonl(p/'tasks',[]);jsonl(p/'predictions',[])
            with self.assertRaisesRegex(ValueError,'no actual task'):
                score(p/'tasks',p/'predictions','none','none','none',p/'out')
            self.assertFalse((p/'out').exists())

    def test_external_review_requires_human_and_hash(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);save(p/'receipt',{'reviewer_type':'model','independent':True})
            with self.assertRaises(ValueError):certify('none','none','none',p/'receipt',p/'out')

    def test_future_addition_and_removal_math(self):
        f0=928/1908
        self.assertAlmostEqual(f0/2,.2431865828092243)
        self.assertAlmostEqual(1-f0/2,.7568134171907757)
        # +10pp enrichment at only 70% FP removal still lowers F1.
        self.assertLess(2*(464-30)/(1908-100),f0)

    def test_cache_boundary_selection_is_invariant_to_gold(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);r=p/'ref';base=r/'experiments/jev';(base/'round4/suites').mkdir(parents=True)
            def put(path,x):path.write_text(json.dumps(x))
            put(base/'round4/selection.json',{'selection_uses_labels':False,'dataset_revision':'fixture',
                                             'document_ids':['1_synthetic','2_excluded']})
            put(base/'document-selection.json',{'confirmation':{'docs':[{'id':'2_excluded'}]}})
            (base/'round3').mkdir()
            put(base/'round3/round3-selection.json',{'test_s40':[],'holdout_rest':[]})
            put(p/'reserve',{'remaining_unused_reserve':[],'reserved_for_j3_confirmation':[]})
            suite={'id':'1_synthetic','tokens':['Fire','smoke'],'mentions':['1','2'],'spans':[[0],[1]],
                   'sentences':[[0,2]],'pair_list':[['1','2'],['2','1']], 'text':'<1 Fire> <2 smoke>', 'gold':'NEVER READ'}
            put(base/'round4/suites/1_synthetic.json',{'documents':[suite]})
            signals=[{'doc':'1_synthetic','a':a,'b':b,'luna':1,'d1':.8,'gold':True}
                     for a,b in [('1','2'),('2','1')]]
            put(base/'round4/controls.json',signals)
            convert(r,p/'reserve',p/'out1')
            for row in signals:row['gold']={'poison':'not a usable label'}
            put(base/'round4/controls.json',signals)
            convert(r,p/'reserve',p/'out2')
            self.assertEqual((p/'out1/baseline.jsonl').read_bytes(),(p/'out2/baseline.jsonl').read_bytes())
            self.assertEqual((p/'out1/selection.json').read_bytes(),(p/'out2/selection.json').read_bytes())
            pilot.parse_docs(p/'out1/baseline.jsonl',{'dev'})


if __name__ == '__main__':
    unittest.main()
