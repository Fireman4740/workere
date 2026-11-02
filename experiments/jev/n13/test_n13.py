"""No network. Local-only transport, safety gates, one-vote/read and synthetic stage."""
import contextlib,copy,io,json,pathlib,tempfile,threading,unittest
from unittest.mock import patch
from pipelines.jev import n13 as n

class Contract(unittest.TestCase):
    def test_no_external_provider_or_fallback(self):
        http=n.LocalHTTP('synthetic-credential')
        for url in ['https://openrouter.ai/api/v1/chat/completions','http://localhost:18020/v1/models','http://127.0.0.1:8000/v1/models']:
            with self.assertRaises(ValueError):http('POST',url,{})
    def test_frozen_design_counts(self):
        spec,rs=n.preflight();self.assertEqual(len(rs),900);self.assertEqual(spec['request_counts'],{'smoke':72,'main':828})
        docs=n.read(n.P/'ctb50-public.json');self.assertEqual(len(docs),50);self.assertEqual(len({int(d['idx'])%10 for d in docs}),10)
    def test_reverse_duplicates_one_voice(self):
        r={'id':'synthetic','model':'qwen3.8-27b','choices':[{'finish_reason':'stop','message':{'content':'[{"pair":"1,2","label":"causes"},{"pair":"2,1","label":"causedby"},{"pair":"1,2","label":"causes"}]'}}]}
        self.assertEqual(n.verify_response(r,{('1','2'),('2','1')}),{('1','2')})
        r['choices'][0]['finish_reason']='length'
        with self.assertRaises(AssertionError):n.verify_response(r,{('1','2'),('2','1')})
    def test_input_token_limits_block_before_generation(self):
        spec,rows=n.preflight();calls=[]
        def fake(method,url,payload=None,timeout=60):calls.append(url);return {'count':16385}
        with self.assertRaises(AssertionError):n.tokenize(fake,spec,rows[0])
        self.assertEqual(len(calls),1);self.assertTrue(calls[0].endswith('/tokenize'))
    def test_readiness_retries_only_model_metadata(self):
        spec,_=n.preflight();calls=[]
        def fake(method,url,payload=None,timeout=60):
            calls.append((method,url))
            if len(calls)==1:raise n.urllib.error.URLError('synthetic loading')
            return {'data':[{'id':'qwen3.8-27b','max_model_len':spec['max_context_tokens']}]}
        with tempfile.TemporaryDirectory() as d,patch.object(n.time,'sleep') as sleep,contextlib.redirect_stdout(io.StringIO()):
            row=n.ready_model(fake,spec,pathlib.Path(d));self.assertEqual(row['id'],'qwen3.8-27b');self.assertEqual(sleep.call_count,1)
        self.assertTrue(all(method=='GET' and url.endswith('/models') for method,url in calls))
    def test_old_or_unreported_context_blocks_before_generation(self):
        spec,_=n.preflight()
        for context in (49152,None):
            row={'id':spec['model']}
            if context is not None:row['max_model_len']=context
            def fake(method,url,payload=None,timeout=60):
                self.assertEqual(method,'GET');self.assertTrue(url.endswith('/models'))
                return {'data':[row]}
            with tempfile.TemporaryDirectory() as d,self.assertRaisesRegex(AssertionError,'context below'):
                n.ready_model(fake,spec,pathlib.Path(d))
    def test_ctb_candidate_scope_and_fold_exclusion(self):
        spec,rows=n.preflight();docs={d['id']:d for d in n.read(n.P/'ctb50-public.json')};bank={d['id']:d for d in n.read(n.P/'demo-bank.json')['documents']}
        self.assertFalse(set(docs)&set(bank))
        for d in docs.values():
            for a,b in d['pair_list']:self.assertEqual(d['sentence_of'][a],d['sentence_of'][b])
        grouped={}
        for r in rows:
            self.assertNotEqual(bank[r['demo']]['fold'],int(docs[r['doc']]['idx'])%10)
            grouped.setdefault((r['doc'],r['schedule'],r['slot']),[]).append(r)
            for mid in docs[r['doc']]['mentions']:self.assertIn('<'+mid+' ',r['body']['messages'][-1]['content'])
        for rs in grouped.values():
            self.assertEqual(len(rs),3)
            self.assertEqual(len({json.dumps(r['body']['messages'][:3],sort_keys=True) for r in rs}),1)
    def test_final50_combines_both_stages_without_extra_generations(self):
        spec,rows=n.preflight();docs=n.read(n.P/'ctb50-public.json');events=[]
        for i,r in enumerate(rows):
            events.append({'logical_id':r['logical_id'],'request':r['body'],'error':None,'response':{
                'id':'synthetic-'+str(i),'model':'qwen3.8-27b','choices':[{'finish_reason':'stop','message':{'content':'[]'}}]}})
        decisions=n.decisions_before_gold(rows,events,docs)
        expected=sum(sum(a<b for a,b in d['pair_list']) for d in docs)*6
        self.assertEqual(len(decisions),expected);self.assertEqual(len({r['doc'] for r in decisions}),50)
        self.assertTrue(all(not r['positive'] for r in decisions))
        with self.assertRaises(AssertionError):n.decisions_before_gold(rows,events[:-1],docs)
        events[-1]['response']['id']=events[0]['response']['id']
        with self.assertRaises(AssertionError):n.decisions_before_gold(rows,events,docs)
    def test_synthetic_stage_freezes_before_gold(self):
        spec,allrows=n.preflight();first=next(r['doc'] for r in allrows if r['stage']=='smoke');rows=[r for r in allrows if r['stage']=='smoke' and r['doc']==first]
        self.assertEqual(len(rows),18)
        doc=next(d for d in n.read(n.P/'smoke-public.json') if d['id']==rows[0]['doc'])
        sequence=iter(range(100));lock=threading.Lock();counts={'tokenize':0,'generation':0}
        def fake(method,url,payload=None,timeout=60):
            with lock:
                if url.endswith('/tokenize'):counts['tokenize']+=1;return {'count':200}
                counts['generation']+=1;rid=next(sequence)
            return {'id':'synthetic-'+str(rid),'model':'qwen3.8-27b','choices':[{'finish_reason':'stop','message':{'content':'[]'}}],'usage':{'prompt_tokens':200,'completion_tokens':2}}
        oldread=n.read
        with tempfile.TemporaryDirectory() as d:
            root=pathlib.Path(d);p=root/'data';p.mkdir()
            n.save(p/'smoke-public.json',[doc]);n.save(p/'smoke-gold.json',[{'doc':doc['id'],'a':a,'b':b,'gold':False} for a,b in doc['pair_list'] if a<b])
            def checked_read(path):
                if str(path).endswith('-gold.json'):self.assertTrue((root/'smoke/freeze-before-gold.json').exists())
                return oldread(path)
            with patch.object(n,'P',p),patch.object(n,'read',checked_read),contextlib.redirect_stdout(io.StringIO()):
                self.assertTrue(n.stage_run(spec,rows,root,fake))
            self.assertEqual(counts,{'tokenize':18,'generation':18})
            self.assertEqual(oldread(root/'smoke/result.json')['new_external_API_usd'],0)
            self.assertTrue((root/'smoke/gold-evaluated.json').exists())

if __name__=='__main__':unittest.main()
