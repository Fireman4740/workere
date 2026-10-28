import copy,json,unittest
from pathlib import Path
from verify import anchored,decode,quote_valid
TEXT='Alice <1 fell> because Bob <2 pushed> her. Alice <3 fell> years earlier.'
DOC=dict(text=TEXT,mentions=[dict(id=x) for x in ['1','2','3']])
def item():
 return dict(candidate_id='c',verdict='SUPPORTED',cause_id='2',effect_id='1',evidence_quote=TEXT.split(' Alice')[0],bindings=[dict(mention_id=x,occurrence_quote=q,participant='Alice' if x=='1' else 'Bob',incident='current fall',time='current',modality='realized',scope_consistent=True) for x,q in [('1','Alice <1 fell>'),('2','Bob <2 pushed>')]])
def raw(i=None,finish='stop',model='pinned'):
 return dict(response=dict(status_code=200,body=dict(model=model,choices=[dict(finish_reason=finish,message=dict(content=json.dumps({'items':[i or item()]})))])))
class Test(unittest.TestCase):
 def test_accept_exact_binding(self):self.assertEqual(anchored(item(),DOC,['1','2']),(True,'accepted'))
 def test_wrong_occurrence_rejected(self):
  i=item();i['bindings'][0]['occurrence_quote']='Alice <3 fell>';self.assertFalse(anchored(i,DOC,['1','2'])[0])
 def test_invented_evidence_rejected(self):
  i=item();i['evidence_quote']='Bob caused the fall';self.assertFalse(anchored(i,DOC,['1','2'])[0])
 def test_missing_scope_rejected(self):
  i=item();i['bindings'][0]['scope_consistent']=False;self.assertFalse(anchored(i,DOC,['1','2'])[0])
 def test_wrong_direction_members_rejected(self):
  i=item();i['cause_id']='3';self.assertFalse(anchored(i,DOC,['1','2'])[0])
 def test_unknown_abstains(self):
  i=item();i['verdict']='UNKNOWN';self.assertEqual(anchored(i,DOC,['1','2']),(False,'not_supported'))
 def test_truncated_quarantined_even_valid_json(self):
  r=decode(dict(custom_id='q',doc_id='d',candidates=[dict(candidate_id='c',pair=['1','2'])]),raw(finish='length'),DOC,dict(allowed_served_models=['pinned']));self.assertFalse(r['valid']);self.assertEqual(r['items'],[]);self.assertEqual(r['state'],'truncated')
 def test_model_guard(self):
  r=decode(dict(custom_id='q',doc_id='d',candidates=[]),raw(model='wrong'),DOC,dict(allowed_served_models=['pinned']));self.assertEqual(r['state'],'model_mismatch')
 def test_coverage_wrong_quarantined(self):
  r=decode(dict(custom_id='q',doc_id='d',candidates=[dict(candidate_id='other',pair=['1','2'])]),raw(),DOC,dict(allowed_served_models=['pinned']));self.assertEqual(r['state'],'coverage_invalid')
 def test_real_plan_gates(self):
  frozen=Path(__file__).parent/'frozen';p=json.loads((frozen/'protocol.json').read_text());rows=json.loads((frozen/'requests.json').read_text());ref=json.loads((frozen/'reference.json').read_text());channels=json.loads((frozen/'channels.json').read_text())
  self.assertLessEqual(p['conservative_api_ceiling_usd'],2);expected={(r['doc_id'],*sorted(r['pair'])) for rr in channels.values() for r in rr};actual={(r['doc_id'],*c['pair']) for r in rows for c in r['candidates']};self.assertEqual(expected,actual)
  self.assertEqual(sum(len(r['candidates']) for r in rows),len(expected))
  for row in rows:
   user=json.loads(row['body']['messages'][1]['content']);self.assertEqual(set(user),{'document_id','tagged_document','candidates'});self.assertLessEqual(len(row['candidates']),3)
   for c in row['candidates']:
    rr=next(v for v in ref[row['doc_id']] if sorted(v['pair'])==c['pair']);self.assertFalse(rr['baseline_positive']);self.assertTrue(rr['luna_zero'])
if __name__=='__main__':unittest.main()
