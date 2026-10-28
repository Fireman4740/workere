import json,tempfile,unittest
from pathlib import Path
from controller import execute,validate,digest,save,canonical
class Test(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.p=dict(expected_served='pinned',per_request_reservation_usd=.0025);self.row=dict(custom_id='q',doc_id='d',pair=['a','b'],arm='generic',body=dict(model='route',state='text',questions={'causal':dict(type='noul')}))
 def tearDown(self):self.tmp.cleanup()
 def transport(self,*args):return dict(model='pinned',answers={'causal':dict(type='noul',noul=.7)},usage=dict(cost=.0001))
 def test_valid_response_cached_no_second_post(self):
  calls=[]
  def call(*args):calls.append(args);return self.transport()
  first=execute(self.row,self.root,self.p,call);second=execute(self.row,self.root,self.p,call)
  self.assertTrue(first['valid']);self.assertEqual(first,second);self.assertEqual(len(calls),1)
 def test_before_post_receipt_exists(self):
  def call(*args):
   record=json.loads((self.root/'q.json').read_text());self.assertEqual(record['state'],'SUBMITTING');return self.transport()
  self.assertTrue(execute(self.row,self.root,self.p,call)['valid'])
 def test_uncertain_prior_post_never_repeated(self):
  save(self.root/'q.json',dict(state='SUBMITTING',request_sha256=digest(self.row['body']),custom_id='q'))
  def forbidden(*args):raise AssertionError('duplicate POST')
  r=execute(self.row,self.root,self.p,forbidden);self.assertFalse(r['valid']);self.assertEqual(r['error'],'ambiguous_prior_POST_never_repeated')
 def test_transport_failure_never_retried(self):
  calls=[]
  def fail(*args):calls.append(1);raise TimeoutError('timeout')
  a=execute(self.row,self.root,self.p,fail);b=execute(self.row,self.root,self.p,fail);self.assertFalse(a['valid']);self.assertEqual(len(calls),1);self.assertEqual(a,b)
 def test_wrong_model_blocks(self):
  with self.assertRaises(ValueError):validate(dict(model='wrong',answers={}),self.p['expected_served'])
 def test_probability_strict(self):
  for value in [True,float('nan'),float('inf'),-.1,1.1,None,'0.5']:
   with self.assertRaises(ValueError):validate(dict(model='pinned',answers={'causal':dict(type='noul',noul=value)}),'pinned')
 def test_cost_overrun_invalid(self):
  def over(*args):r=self.transport();r['usage']['cost']=.003;return r
  r=execute(self.row,self.root,self.p,over);self.assertFalse(r['valid']);self.assertEqual(r['state'],'COST_BOUND_EXCEEDED')
 def test_frozen_plan_pairs_budget_and_matched_context(self):
  f=Path(__file__).parent/'frozen';p=json.loads((f/'protocol.json').read_text());public=json.loads((f/'public.json').read_text());rows=public['requests'];self.assertEqual(len(rows),146);self.assertLessEqual(len(rows)*p['per_request_reservation_usd'],p['cap_usd'])
  groups={}
  for r in rows:
   self.assertGreaterEqual(r['input_utf8_bound'],len(canonical(r['body']).encode()));groups.setdefault((r['doc_id'],*r['pair']),{})[r['arm']]=r
  self.assertEqual(len(groups),73)
  for group in groups.values():self.assertEqual(set(group),{'generic','lexical'});self.assertEqual(group['generic']['body']['state'],group['lexical']['body']['state'])
if __name__=='__main__':unittest.main()
