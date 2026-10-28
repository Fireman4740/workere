import copy, json, unittest
from experiments.jev.n2_transport.n2 import *

class N2Tests(unittest.TestCase):
 def setUp(self):
  self.b,self.rows,self.p=prepare();self.row=self.rows[0];self.c=self.row['extraction']['certificate'];self.doc=self.b['documents'][self.row['root']['doc']]
  self.config=read_json(PLAN/'config.json');self.req=request('source',self.doc,{'item_id':self.row['root']['id'],'certificate':self.c,'binding_mentions':[self.c['cause'],self.c['effect']]},{'test':1},self.config)
  sids=sorted({self.doc['mentions'][m]['sentence'] for m in [self.c['cause'],self.c['effect']]})
  self.raw={'item_id':self.row['root']['id'],'checks':{k:'SUPPORTED' for k in CHECKS['source']},'evidence_sentences':sids,'reason':'fixture evidence'}
 def receipt(self,raw):return {'response':{'items':[raw]}}
 def test_supported_requires_exact_mention_sentence(self):
  raw=copy.deepcopy(self.raw);raw['evidence_sentences']=[]
  self.assertFalse(validate_answer(self.req,self.receipt(raw),self.b)['valid'])
 def test_unknown_is_valid_abstention(self):
  raw=copy.deepcopy(self.raw);raw['checks']['causal_source']='UNKNOWN';raw['evidence_sentences']=[]
  result=validate_answer(self.req,self.receipt(raw),self.b);self.assertTrue(result['valid']);self.assertFalse(result['supported'])
 def test_truncation_quarantined(self):
  self.assertFalse(validate_answer(self.req,{'execution_error':'TRUNCATED','response':{}},self.b)['valid'])
 def test_unknown_sentence_rejected(self):
  raw=copy.deepcopy(self.raw);raw['evidence_sentences']=[9999]
  self.assertFalse(validate_answer(self.req,self.receipt(raw),self.b)['valid'])
 def test_quotes_computed_exact(self):
  result=validate_answer(self.req,self.receipt(self.raw),self.b);self.assertTrue(result['supported'])
  for c in result['citations']:self.assertEqual(c['quote'],self.doc['text'][c['start']:c['end']])
 def test_candidates_fixed_and_initially_negative(self):
  self.assertEqual(digest(self.p),read_json(HERE/'protocol.json')['planned_registry_hash'])
  for p in self.p:
   scores={tuple(r['pair']):r['fusion'] for r in self.b['scores'][p['doc']]}
   self.assertLess(scores[tuple(p['target'])],.39);self.assertNotEqual(p['cause'],p['alternative'])
 def test_no_extra_or_duplicate_item(self):
  r=self.receipt(self.raw);r['response']['items'].append(copy.deepcopy(self.raw))
  self.assertFalse(validate_answer(self.req,r,self.b)['valid'])
if __name__=='__main__':unittest.main()
