"""No API: meaningful prefix, duplicate-label, global validity and NI checks."""
import base64,contextlib,hashlib,importlib.util,io,json,tempfile,unittest,zipfile
from pathlib import Path
spec=importlib.util.spec_from_file_location('n8',Path(__file__).with_name('n8.py'));n=importlib.util.module_from_spec(spec);spec.loader.exec_module(n)
def response(content,finish='stop'):return {'model':'openai/gpt-6-luna','provider':'OpenAI','choices':[{'finish_reason':finish,'message':{'content':content}}]}
class Contract(unittest.TestCase):
 def test_last_label_wins_and_unknown_blocks(self):
  allowed={('1','2')};self.assertEqual(n.positive_pairs(response('[{"pair":"1,2","label":"causes"},{"pair":"1,2","label":"norel"}]'),allowed),set())
  with self.assertRaises(ValueError):n.positive_pairs(response('[{"pair":"9,2","label":"causes"}]'),allowed)
  with self.assertRaises(ValueError):n.positive_pairs(response('[]','length'),allowed)
  bad=response('[]');bad['provider']='other'
  with self.assertRaises(ValueError):n.positive_pairs(bad,allowed)
  bad=response('[]');bad['choices']*=2
  with self.assertRaises(ValueError):n.positive_pairs(bad,allowed)
 def test_none_response_keeps_financial_report_and_export(self):
  events=[{'batch_id':'batch1','provider_reported_batch_cost_usd':.01,'response':None}]
  report=n.financial_summary(events);self.assertEqual(report['new_API_reported_cost_USD'],.01);self.assertEqual(report['usage_totals']['completion_tokens'],0)
  original=n.ROOT,n.HERE,n.STATE
  with tempfile.TemporaryDirectory() as t:
   n.ROOT=Path(t);n.HERE=Path(t)/'experiments/jev/n8';n.HERE.mkdir(parents=True);n.STATE=Path(t)/'logs/n8';n.STATE.mkdir(parents=True)
   (Path(t)/'experiments/jev/run.sh').write_text('true');(n.HERE/'protocol.json').write_text('{}');(n.STATE/'events.jsonl').write_text(json.dumps(events[0])+'\n');(n.STATE/'summary.json').write_text(json.dumps(report))
   try:
    capture=io.StringIO()
    with contextlib.redirect_stdout(capture):n.export()
    lines=capture.getvalue().splitlines();chunks=[json.loads(line.split(' ',1)[1]) for line in lines if line.startswith('N8_EVIDENCE_CHUNK ')];end=json.loads(lines[-1].split(' ',1)[1]);data=base64.b64decode(''.join(c['data'] for c in chunks));self.assertEqual(hashlib.sha256(data).hexdigest(),end['sha256'])
    with zipfile.ZipFile(io.BytesIO(data)) as z:self.assertIn('logs/n8/events.jsonl',z.namelist())
   finally:n.ROOT,n.HERE,n.STATE=original
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
