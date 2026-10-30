"""No API. Representation and logical-generation integrity plus synthetic integration."""
import contextlib
import copy
import io
import json
import os
import pathlib
import shutil
import tempfile
import unittest
from unittest.mock import patch
from pipelines.jev import n12 as n
from pipelines.jev.n12_transport import FreshLogicalBatchClient

class Contract(unittest.TestCase):
    def test_views_preserve_ids_and_exact_original(self):
        for d in n.read(n.P/'public.json'):
            texts,info=n.views(d)
            self.assertEqual(texts['original'],d['text'])
            self.assertTrue(texts['table'].startswith(d['text']))
            for mid in d['mentions']:
                self.assertIn('<'+mid+' ',texts['local'])
            self.assertTrue(info['all_mentions_retained'])
    def test_pairing_fresh_slots_and_conservative_bounds(self):
        rs=n.prepare()
        self.assertEqual(len(rs),72)
        self.assertEqual(len({r['custom_id'] for r in rs}),72)
        for r in rs:
            self.assertLessEqual(r['input_utf8_bound']+32768,65536)
            self.assertNotEqual(r['doc'].split('_')[0],r['demo'].split('_')[0])
    def test_invalid_response_blocks_global_scoring(self):
        response={'model':'openai/gpt-6-luna','provider':'OpenAI','choices':[{'finish_reason':'length','message':{'content':'[]'}}]}
        with self.assertRaises(AssertionError):n.validate_content(response,{('1','2')})
        response['choices'][0]['finish_reason']='stop';response['choices'][0]['message']['content']='[{"pair":"999,2","label":"causes"}]'
        with self.assertRaises(AssertionError):n.validate_content(response,{('1','2')})
    def test_synthetic_end_to_end_no_network(self):
        actual=n.P;posts=[]
        def fake_http(client,method,url,payload=None):
            self.assertEqual(method,'POST');posts.append(payload)
            return {'id':'synthetic-'+payload['requests'][0]['custom_id'],'status':'completed','usage':{'cost':0},
                    'results':[{'custom_id':r['custom_id'],'response':{'status_code':200,'body':{
                        'id':'response-'+r['custom_id'],'model':'openai/gpt-6-luna','provider':'OpenAI',
                        'choices':[{'finish_reason':'stop','message':{'content':'[]'}}],
                        'usage':{'prompt_tokens':0,'completion_tokens':0}}}} for r in payload['requests']]}
        with tempfile.TemporaryDirectory() as tmp:
            tmp=pathlib.Path(tmp);p=tmp/'fixture';p.mkdir()
            for name in ['public.json','protocol.json','selection.json','preflight-frozen.json','requests-frozen.json','test_n12.py']:
                shutil.copyfile(actual/name,p/name)
            controls=[{'doc':d['id'],'a':a,'b':b,'d1':0.,'gold':False} for d in n.read(actual/'public.json') for a,b in d['pair_list']]
            n.save(p/'controls.json',controls);n.save(p/'d1-public.json',[{k:r[k] for k in ('doc','a','b','d1')} for r in controls])
            run='synthetic-n12-test';out=tmp/'logs/n12'/run
            # main uses relative logs; change only pathlib's construction at its one controller point.
            source=pathlib.Path(n.__file__).read_text().replace('pathlib.Path("logs/n12") / run','pathlib.Path('+repr(str(tmp/'logs/n12'))+') / run')
            module={**n.__dict__};exec(compile(source,n.__file__,'exec'),module);module['P']=p
            with patch.object(FreshLogicalBatchClient,'_http',fake_http),patch.dict(os.environ,{'ORX_RUN_ID':run,'ORX_COMMIT':'synthetic'}),contextlib.redirect_stdout(io.StringIO()):
                module['catalog']=lambda root:n.save(root/'catalog-response.json',{'synthetic':True})
                module['main']()
            self.assertEqual(len(posts),4)
            self.assertEqual(sum(len(p['requests']) for p in posts),72)
            result=n.read(out/'result.json');self.assertEqual(result['status'],'COMPLETE_EXPLORATORY_SMOKE')
            self.assertEqual(result['cost_known_usd'],0)
            self.assertTrue((out/'freeze-before-gold.json').exists())
            self.assertEqual(len({r['custom_id'] for p in posts for r in p['requests']}),72)

if __name__=='__main__':unittest.main()
