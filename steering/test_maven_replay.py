import itertools
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from steering import campaign_api as api
from steering import maven_budget as mb
from steering import maven_replay as r


def public():
    return {'d': dict(id='d',text='The <e1 blast> led to the <e2 collapse>.',
                      pair_list=[['e1','e2'],['e2','e1']])}


def bases():
    return [dict(doc_id='d',pass_id=i,demo_id='train'+str(i),messages=[
        dict(role='system',content='MAVEN rules including implicit causes.'),
        dict(role='user',content='demo'+str(i)),dict(role='assistant',content='demo labels'),
        dict(role='user',content='Document:\n'+public()['d']['text'])]) for i in range(5)]


class Transfers(unittest.TestCase):
    def test_diversity_and_sampling_have_exact_five_read_contract(self):
        source=bases()
        greedy=r.document_jobs(source)
        sample=r.document_jobs(source,True)
        self.assertEqual(len(greedy),5)
        self.assertEqual(len({x['demo_id'] for x in greedy}),5)
        self.assertEqual({x['temperature'] for x in greedy},{0})
        self.assertEqual(len({x['demo_id'] for x in sample}),1)
        self.assertEqual(len({x['seed'] for x in sample}),5)
        self.assertEqual({(x['thinking'],x['answer_tokens']) for x in sample},{(8192,4096)})
        self.assertEqual(source,bases())

    def test_document_invalid_empty_and_outside_universe_removed(self):
        self.assertEqual(r.document_pairs(dict(object=None),public()['d']),set())
        obj=dict(relations=[dict(pair='e2,e1',label='causes'),dict(pair='missing,e1',label='causes')])
        self.assertEqual(r.document_pairs(dict(object=obj),public()['d']),{('e1','e2')})

    def test_n2_fixed_denominator_and_maven_semantics(self):
        rows=[dict(doc_id='d',object=None) for _ in range(5)]
        rows[0]['object']=dict(relations=[dict(pair='e1,e2',label='causes')])
        result=r.verification_jobs(bases(),rows,public())
        self.assertEqual(len(result),1)
        note=result[0]['messages'][-1]['content']
        self.assertIn('1/5 readings',note)
        self.assertIn('implicit causation is allowed',note)
        self.assertNotIn('required lexical anchor',note)
        with self.assertRaises(ValueError): r.verification_jobs(bases(),rows[:4],public())

    def test_screens_cover_all_pairs_once_per_order(self):
        jobs=r.screen_jobs(public(),'rules')
        self.assertEqual(len(jobs),2)
        self.assertEqual({x['order'] for x in jobs},{0,1})
        self.assertEqual(jobs[0]['pairs'],jobs[1]['pairs'])
        self.assertEqual({(x['thinking'],x['answer_tokens']) for x in jobs},{(4096,384)})
        rows=[dict(object={'0':'N'}),dict(object=None)]
        views=r.views(jobs,rows)
        base={'d':{('e1','e2')}}
        self.assertEqual(r.local_filter(base,views),base)
        rows[1]['object']={'0':'N'}
        self.assertEqual(r.local_filter(base,r.views(jobs,rows)),{'d':set()})

    def test_single_reject_and_one_positive_cannot_remove(self):
        base={'d':{('e1','e2')}}
        for left,right in itertools.product((None,True,False),repeat=2):
            views={('d',i):({} if value is None else {('e1','e2'):value})
                   for i,value in enumerate((left,right))}
            chosen=('e1','e2') in r.local_filter(base,views)['d']
            self.assertEqual(chosen,not(left is False and right is False))

    def test_routing_only_valid_disagreement(self):
        base={'d':{('e1','e2')}}
        self.assertEqual(r.direct_jobs(public(),'rules',base,{})[0],[])
        jobs,routes=r.direct_jobs(public(),'rules',base,{('d',0):{('e1','e2'):False}})
        self.assertEqual(len(jobs),2)
        self.assertEqual(len(routes),1)
        self.assertEqual({x['reverse'] for x in jobs},{False,True})
        self.assertNotIn('gold',api.canonical(jobs))
        self.assertEqual({(x['thinking'],x['answer_tokens']) for x in jobs},{(2560,512)})

    def test_asymmetric_complete_and_missing_truth_table(self):
        for checks in itertools.product((True,False),repeat=4):
            self.assertEqual(r.asymmetric(False,list(checks)),sum(checks)==4)
            self.assertEqual(r.asymmetric(True,list(checks)),sum(checks)>1)
        for checks in itertools.product((True,False,None),repeat=4):
            if None in checks:
                self.assertTrue(r.asymmetric(True,list(checks)))
                self.assertFalse(r.asymmetric(False,list(checks)))

    def test_w4_orders_and_cap_are_gold_blind(self):
        ids=[f'e{i}' for i in range(30)]
        ps={(ids[0],x) for x in ids[1:]}
        data={'d':dict(id='d',text=' '.join(f'<{x} event>' for x in ids),pair_list=sorted(ps))}
        jobs,routes=r.w4_jobs(data,'rules',{'d':ps},{'d':{p:3 for p in ps}})
        self.assertEqual(len(routes['d']['selected']),24)
        self.assertEqual(len(jobs),6)
        for left,right in zip(jobs[::2],jobs[1::2]):
            self.assertFalse(left['reverse']); self.assertTrue(right['reverse'])
            self.assertEqual(left['pairs'],right['pairs'])
            self.assertEqual(len(left['pairs']),8)
            p=left['pairs'][0]
            self.assertIn(f'A=<{p[1]}>, B=<{p[0]}>',right['messages'][-1]['content'])

    def test_freeze_roundtrip_does_not_rewrite_but_detects_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'pairs.json'
            r.freeze(path,{'pairs':[('a','b')]})
            before=(path.read_bytes(),path.stat().st_mtime_ns)
            r.freeze(path,{'pairs':[('a','b')]})
            self.assertEqual(before,(path.read_bytes(),path.stat().st_mtime_ns))
            with self.assertRaises(ValueError): r.freeze(path,{'pairs':[('a','c')]})

    def test_response_contract_and_content_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp); req=dict(key='k'); manifest={'panel':'frozen'}
            data=dict(object=None)
            path=out/'responses'/'k.json'
            mb.atomic(path,dict(contract=api.digest([manifest,req]),sha256=api.digest(data),data=data))
            self.assertEqual(r.checked_rows(out,manifest,[req]),[data])
            with self.assertRaises(ValueError): r.checked_rows(out,{'panel':'changed'},[req])
            row=mb.read(path); row['data']['object']={'label':'A'}; mb.atomic(path,row)
            with self.assertRaises(ValueError): r.checked_rows(out,manifest,[req])

    def test_checkpoint_never_calls_model_after_deadline(self):
        with tempfile.TemporaryDirectory() as tmp:
            class Fake:
                out=Path(tmp)
                manifest={}
                cfg=dict(arm='N1',engine=dict(concurrency=2))
                def check(self,jobs): pass
                def one(self,job): raise AssertionError('model must not be called')
            jobs=[dict(key='unstarted')]
            self.assertFalse(r.phase(Fake(),'test',jobs,time.monotonic()-1))
            self.assertFalse((Path(tmp)/'responses').exists())

    def test_stage_resume_reuses_thinking_and_completed_answer(self):
        class Tokenizer:
            def apply_chat_template(self,*args,**kwargs): return 'prompt'
            def encode(self,text,**kwargs): return list(text.encode())
        class Client:
            def __init__(self): self.calls=[]
            def call(self,path,body):
                self.calls.append(body)
                final='structured_outputs' in body
                return {'choices':[dict(index=0,text='{"relations":[]}' if final else 'reason',
                    finish_reason='stop',stop_reason='</think>',token_ids=[1])]},False,.1
        with tempfile.TemporaryDirectory() as tmp, patch.dict('os.environ',{'ORX_RUN_ID':'run','ORX_COMMIT':'commit'}):
            reader=r.Reader.__new__(r.Reader)
            reader.out=Path(tmp); reader.manifest={'frozen':True}
            reader.cfg=dict(model='test',engine=dict(max_model_len=65536))
            reader.tok=Tokenizer(); reader.client=Client()
            req=r.document_jobs(bases())[0]
            row=reader.one(req)
            self.assertEqual(row['object'],dict(relations=[]))
            self.assertEqual(len(reader.client.calls),2)
            self.assertEqual(reader.one(req),row)
            self.assertEqual(len(reader.client.calls),2)
            (reader.out/'responses'/(req['key']+'.json')).unlink()
            self.assertEqual(reader.one(req),row)
            self.assertEqual(len(reader.client.calls),3)
            self.assertIn('structured_outputs',reader.client.calls[-1])


if __name__=='__main__': unittest.main()
