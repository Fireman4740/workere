"""No API: meaningful prefix, duplicate-label, global validity and NI checks."""
import importlib.util,json,unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('n8',Path(__file__).with_name('n8.py'));n=importlib.util.module_from_spec(spec);spec.loader.exec_module(n)
def response(content,finish='stop'):return {'model':'openai/gpt-6-luna','choices':[{'finish_reason':finish,'message':{'content':content}}]}
class Contract(unittest.TestCase):
 def test_last_label_wins_and_unknown_blocks(self):
  allowed={('1','2')};self.assertEqual(n.positive_pairs(response('[{"pair":"1,2","label":"causes"},{"pair":"1,2","label":"norel"}]'),allowed),set())
  with self.assertRaises(ValueError):n.positive_pairs(response('[{"pair":"9,2","label":"causes"}]'),allowed)
  with self.assertRaises(ValueError):n.positive_pairs(response('[]','length'),allowed)
 def test_shared_prefix_not_best_subset(self):
  pub=[{'id':str(d)+'_doc','candidates':[['1','2'],['2','1']]} for d in range(8)];rs=[];es=[]
  for d in pub:
   for i in range(7):
    messages=[{'role':'user','content':d['id']+str(i)}];rs.append({'doc':d['id'],'pass_index':i,'messages':messages})
    es.append({'request':{'messages':messages,'max_tokens':32768,'reasoning':{'effort':'high'},'model':'openai/gpt-6-luna:batch'},'batch_api':True,'response':response('[]' if i<3 else '[{"pair":"1,2","label":"causes"}]')})
  rows=n.predictions_from_events(rs,es,pub);self.assertTrue(all(r['L3']==0 and (r['L7']==4/7 if r['a']=='1' else r['L7']==0) for r in rows))
  es[-1]['response']=response('[]','length')
  with self.assertRaises(ValueError):n.predictions_from_events(rs,es,pub)
 def test_identical_systems_meet_frozen_margin(self):
  controls=[];preds=[]
  for i in range(8):
   for a,b in [('1','2'),('2','1')]:controls.append({'doc':str(i)+'_doc','a':a,'b':b,'gold':True,'d1':1});preds.append({'doc':str(i)+'_doc','a':a,'b':b,'L3':1,'L7':1})
  out,_=n.evaluate(preds,controls);self.assertEqual(out['topic_bootstrap_CI95'],[0.,0.]);self.assertEqual(out['decision'],'NONINFERIOR_DEVELOPMENT')
if __name__=='__main__':unittest.main()
