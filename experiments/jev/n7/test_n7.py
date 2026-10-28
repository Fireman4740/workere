import json,pathlib,tempfile,unittest
from unittest.mock import patch
from pipelines.jev.n7 import prepare,freeze_predictions,validate_content
from pipelines.jev.batch import BatchChatClient
class N7Tests(unittest.TestCase):
 def test_plan(self):
  req=prepare();self.assertEqual(len(req),192);self.assertEqual(len({r['body_sha256'] for r in req}),135)
  for r in req:
   self.assertNotEqual(r['doc'].split('_')[0],r['demo'].split('_')[0]);self.assertLessEqual(r['input_utf8_bound']+32768,65536)
 def test_invalid_unknown_and_last_vote(self):
  response={'model':'openai/gpt-6-luna','provider':'OpenAI','choices':[{'finish_reason':'stop','message':{'content':'[{"pair":"1,2","label":"causes"},{"pair":"1,2","label":"norel"}]'}}]}
  self.assertEqual(validate_content(response,{('1','2')}),{('1','2'):'norel'})
  with self.assertRaises(AssertionError):validate_content(response,{('3','4')})
  response['choices'][0]['finish_reason']='length'
  with self.assertRaises(AssertionError):validate_content(response,{('1','2')})
 def test_global_gate_blocks_one_bad_event(self):
  req=prepare();events=[]
  for r in req:events.append({'logical_id':r['logical_id'],'cache_hit':False,'error':None,'request':r['body'],'response':{'model':'openai/gpt-6-luna','provider':'OpenAI','choices':[{'finish_reason':'stop','message':{'content':'[]'}}]}})
  predictions,failures=freeze_predictions(req,events);self.assertFalse(failures);self.assertEqual(len(predictions),64)
  events[-1]['response']['choices'][0]['finish_reason']='length';predictions,failures=freeze_predictions(req,events);self.assertIsNone(predictions);self.assertEqual(len(failures),1)
 def test_identical_schedule_prompts_are_new_submissions(self):
  calls=[];events=[]
  def http(client,method,url,payload=None):
   self.assertEqual(method,'POST');calls.append(payload);bid='mock-batch-'+str(len(calls))
   return {'id':bid,'status':'completed','usage':{'cost':0},'results':[{'custom_id':r['custom_id'],'response':{'status_code':200,'body':{'model':'openai/gpt-6-luna','provider':'OpenAI','choices':[{'finish_reason':'stop','message':{'content':'[]'}}]}}} for r in payload['requests']]}
  req=[r for r in prepare() if r['schedule']==0 and r['arm']=='contrastive'][:3]
  with tempfile.TemporaryDirectory() as tmp,patch.object(BatchChatClient,'_http',http):
   for schedule in (0,1):
    client=BatchChatClient(model='openai/gpt-6-luna:batch',model_version='gpt-6-luna-20260922',mode='live',cache_path=f'{tmp}/s{schedule}/cache.sqlite',max_calls=3,budget_usd=.08,max_cost_per_call_usd=.025,input_price_per_million=.1,max_output_tokens=32768,max_context_tokens=65536,retries=0)
    client.decide_many([r['body']['messages'] for r in req],[events.append]*3);client.close()
  self.assertEqual(len(calls),2);self.assertEqual(len(events),6);self.assertTrue(all(not e['cache_hit'] for e in events));self.assertNotEqual(events[0]['batch_id'],events[3]['batch_id'])
if __name__=='__main__':unittest.main()
