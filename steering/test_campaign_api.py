"""CPU-only protocol and failure checks for the API transport."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from steering import campaign_api as api


class Tokenizer:
    def encode(self,text,**kwargs):
        return list(text.encode())

    def apply_chat_template(self,messages,**kwargs):
        return messages[-1]['content']+'<think>'


class ApiTests(unittest.TestCase):
    def test_choice_order_and_incomplete_batch(self):
        result={'choices':[{'index':1,'text':'B'},{'index':0,'text':'A'}]}
        self.assertEqual([x['text'] for x in api.ordered_choices(result,2)],['A','B'])
        for choices in ([{'index':0}],[{'index':0},{'index':0}],[{'index':0},{'index':3}]):
            with self.assertRaises(ValueError): api.ordered_choices({'choices':choices},2)

    def test_two_stage_budgets_are_not_generic_chat_effort(self):
        first=api.payload('qwen',[[1,2]],1024)
        self.assertEqual(first['stop'],['</think>'])
        self.assertFalse(first['add_special_tokens'])
        self.assertNotIn('structured_outputs',first)
        second=api.payload('qwen',[[1,2,3]],512,'evidence')
        self.assertNotIn('stop',second)
        self.assertEqual(second['structured_outputs']['json'],api.a.answer_schema('evidence'))
        self.assertEqual(second['max_tokens'],512)

    def test_batch_splits_on_schema_and_preserves_all_keys(self):
        rows=[{'key':str(i),'arm':a} for i,a in enumerate(['direct']*3+['evidence']*3)]
        chunks=api.batches(rows,2)
        self.assertEqual([len(x) for x in chunks],[2,1,2,1])
        self.assertEqual([r for chunk in chunks for r in chunk],rows)
        self.assertTrue(all(len({r['arm'] for r in chunk})==1 for chunk in chunks))

    def test_final_only_and_length_abstention(self):
        requests=[dict(key='x',dataset='CTB',arm='direct',reverse=False)]
        trace=[dict(text='{"label":"A"}',finish_reason='length',stop_reason=None,token_ids=[1,2])]
        for finish,status in [('length','invalid_answer'),('stop','valid')]:
            final=[dict(text='{"label":"N"}',finish_reason=finish,token_ids=[3])]
            row=api.generation_rows(requests,['p'],trace,final,Tokenizer(),4,{})[0]
            self.assertEqual(row['status'],status)
            self.assertEqual(row['answer'],'{"label":"N"}')
            self.assertTrue(row['thinking_forced'])
            self.assertEqual(row['think_tokens'],2)
            self.assertEqual(row['wall_seconds_share'],4)

    def test_cache_identity_and_path_resolution_do_not_include_auth(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def read(self): return b'{"choices":[]}'
        with tempfile.TemporaryDirectory() as tmp, patch('urllib.request.urlopen',return_value=Response()) as openurl:
            client=api.Client('http://localhost:18020/v1','SECRET',Path(tmp),'server-a')
            body={'prompt':[[1]],'max_tokens':1}
            client.call('/completions',body,cache=True)
            _,hit,_=client.call('/completions',body,cache=True)
            self.assertTrue(hit)
            self.assertEqual(openurl.call_count,1)
            client.call('/../tokenize',{'prompt':'hello'})
            self.assertEqual(openurl.call_args.args[0].full_url,'http://localhost:18020/tokenize')
            self.assertNotIn('SECRET',''.join(p.read_text() for p in Path(tmp).glob('*.json')))
            other=api.Client('http://localhost:18020/v1','SECRET',Path(tmp),'server-b')
            _,hit,_=other.call('/completions',body,cache=True)
            self.assertFalse(hit)

    def test_complete_transport_mock_keeps_order_and_continuation_prefix(self):
        class Client:
            def __init__(self): self.calls=[]
            def call(self,path,body,**kwargs):
                self.calls.append(body)
                if len(self.calls)==1:
                    return {'choices':[dict(index=1,text='second',finish_reason='length'),
                                       dict(index=0,text='first',finish_reason='length')],'usage':{'completion_tokens':2}},False,2
                return {'choices':[dict(index=1,text='{"label":"N"}',finish_reason='stop'),
                                   dict(index=0,text='{"label":"A"}',finish_reason='stop')],
                        'usage':{'completion_tokens':2}},False,1
        reqs=[dict(key=str(i),dataset='CTB',arm='direct',reverse=False,
                   messages=[{'role':'user','content':str(i)}]) for i in range(2)]
        client=Client()
        rows,cost=api.execute_batch(client,Tokenizer(),reqs,
            dict(think_tokens=1024,answer_tokens=512,max_model_len=32768),'qwen')
        self.assertEqual([json.loads(r['answer'])['label'] for r in rows],['A','N'])
        self.assertEqual(bytes(client.calls[1]['prompt'][0]).decode(),'0<think>first\n</think>\n\n')
        self.assertEqual(sum(r['api_batch_usage_share']['completion_tokens'] for r in rows),4)
        self.assertEqual(cost['cache_hits'],0)


if __name__=='__main__': unittest.main()
