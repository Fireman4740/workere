"""Meaningful offline contracts for the new parallel/chunked inference path."""
import contextlib
import copy
import io
import json
import pathlib
import tempfile
import threading
import unittest
from unittest.mock import patch
from pipelines.jev import n14 as n

class N14Contracts(unittest.TestCase):
    def test_every_original_pair_exposed_once_per_read_all_conditions(self):
        spec,rows,docs=n.preflight()
        self.assertEqual(len(rows),1332)
        self.assertEqual(spec['request_counts'],{'M00':72,'M10':72,'M01':594,'M11':594})
        self.assertEqual(sum(len(d['pair_list'])//2 for d in docs),593)
        old={r['logical_id']:r for r in map(json.loads,pathlib.Path('experiments/jev/n13/ctb50/requests.jsonl').read_text().splitlines())}
        for r in rows:
            if r['condition']=='M00':
                original=r['logical_id'].removeprefix('n14:M00:').rsplit(':b',1)[0]
                self.assertEqual(r['body']['messages'],old[original]['body']['messages'])
    def test_chunk_parser_requires_complete_nonduplicated_explicit_coverage(self):
        _,rows,_=n.preflight();r=next(r for r in rows if r['bounded'])
        content=[{'pair':','.join(p),'label':'norel'} for p in r['allowed']]
        response={'id':'synthetic','model':'qwen3.8-27b','choices':[{'finish_reason':'stop','message':{'content':json.dumps(content)}}]}
        self.assertEqual(n.validate(response,r),set())
        for items in (content[:-1],content+[content[0]],[{'pair':'unknown,ids','label':'causes'}]):
            bad=copy.deepcopy(response);bad['choices'][0]['message']['content']=json.dumps(items)
            with self.assertRaises(AssertionError):n.validate(bad,r)
        bad=copy.deepcopy(response);bad['choices'][0]['finish_reason']='length'
        with self.assertRaises(AssertionError):n.validate(bad,r)
    def test_medium_gate_checks_rendered_tokens_before_any_generation(self):
        spec,_,_=n.preflight();calls=[]
        def fake(method,url,body,timeout=60):
            calls.append((method,url));effort=body['chat_template_kwargs']['reasoning_effort']
            return {'count':3,'tokens':[1,2,3 if effort=='medium' else 4]}
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            n.template_gate(fake,spec,pathlib.Path(d))
        self.assertEqual(len(calls),2)
        self.assertTrue(all(url.endswith('/tokenize') for _,url in calls))
        def same(*a,**k):return {'count':3,'tokens':[1,2,3]}
        with tempfile.TemporaryDirectory() as d,self.assertRaisesRegex(AssertionError,'identically'):
            n.template_gate(same,spec,pathlib.Path(d))
    def test_missing_or_invalid_request_never_becomes_negative_vote(self):
        _,rows,docs=n.preflight();doc=next(d for d in docs if d['id']=='wsj_0006')
        rs=[r for r in rows if r['condition']=='M00' and r['doc']==doc['id']]
        def event(r,i):return {'logical_id':r['logical_id'],'request':r['body'],'error':None,'response':{
            'id':'fake'+str(i),'model':'qwen3.8-27b','choices':[{'finish_reason':'stop','message':{'content':'[]'}}]}}
        ev=[event(r,i) for i,r in enumerate(rs)]
        self.assertEqual(n.freeze_decisions('M00',rs,ev,[doc])['complete_documents'],[doc['id']])
        for bad in (ev[:-1],ev[:-1]+[{**ev[-1],'error':{'type':'test'}}]):
            result=n.freeze_decisions('M00',rs,bad,[doc]);self.assertEqual(result['complete_documents'],[]);self.assertEqual(result['decisions'],[])
    def test_synthetic_four_condition_run_freezes_all_before_gold(self):
        spec,allrows,docs=n.preflight();doc=next(d for d in docs if d['id']=='wsj_0006')
        rows=[r for r in allrows if r['doc']==doc['id']];mapping={n.h.sha(n.h.canonical(r['body'])):r for r in rows};lock=threading.Lock();counter=[0]
        def fake(method,url,body,timeout=60):
            if url.endswith('/tokenize'):return {'count':200,'tokens':[1]*200}
            r=mapping[n.h.sha(n.h.canonical(body))]
            content=[{'pair':','.join(p),'label':'norel'} for p in r['allowed']] if r['bounded'] else []
            with lock:counter[0]+=1;rid='synthetic-'+str(counter[0])
            return {'id':rid,'model':'qwen3.8-27b','choices':[{'finish_reason':'stop','message':{'content':json.dumps(content)}}],
                    'usage':{'prompt_tokens':200,'completion_tokens':10,'total_tokens':210}}
        read=n.h.read
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            root=pathlib.Path(d);data=root/'data';data.mkdir();state=root/'state';state.mkdir()
            n.h.save(data/'gold.json',[{'doc':doc['id'],'a':a,'b':b,'gold':False} for a,b in doc['pair_list'] if a<b])
            def checked(path):
                if pathlib.Path(path)==data/'gold.json':self.assertTrue((state/'freeze-before-gold.json').exists())
                return read(path)
            events={c:n.run_condition(c,[r for r in rows if r['condition']==c],fake,spec,state) for c in n.CONDITIONS}
            with patch.object(n,'P',data),patch.object(n.h,'read',checked):result=n.finish(rows,events,[doc],state)
            self.assertEqual(counter[0],len(rows))
            self.assertEqual(result['common_complete_documents'],[doc['id']])
            self.assertTrue(all(not x['invalid_calls'] for x in result['conditions'].values()))
    def test_transport_stays_local_without_fallback(self):
        http=n.h.LocalHTTP('synthetic')
        with self.assertRaises(ValueError):http('POST','https://example.com/v1/chat/completions',{})

if __name__=='__main__':unittest.main()
