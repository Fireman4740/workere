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
        spec,rs=n.preflight();self.assertEqual(len(rs),792);self.assertEqual(spec['request_counts'],{'smoke':72,'pilot40':720})
        docs=n.read(n.P/'pilot40-public.json');self.assertEqual(len(docs),40);self.assertEqual(len({d['id'].split('_')[0] for d in docs}),20)
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
    def test_dependency_waits_for_release(self):
        spec,_=n.preflight()
        with tempfile.TemporaryDirectory() as d,patch.object(n,'active_dependency_processes',side_effect=[[100],[100],[]]),patch.object(n.time,'sleep') as sleep,contextlib.redirect_stdout(io.StringIO()):
            n.wait_for_dependency(spec,pathlib.Path(d));self.assertEqual(sleep.call_count,2)
            self.assertEqual(n.read(pathlib.Path(d)/'queue-status.json')['phase'],'LOCAL_CAPACITY_RELEASED')
    def test_synthetic_stage_freezes_before_gold(self):
        spec,allrows=n.preflight();rows=[r for r in allrows if r['stage']=='smoke' and r['doc']==allrows[0]['doc']]
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
