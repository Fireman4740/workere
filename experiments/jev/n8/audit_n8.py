"""Independent standard-library replay of exported N8 evidence, no network/model."""
import argparse,hashlib,json,random,zipfile
from collections import defaultdict
from pathlib import Path

def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def f1(c):return 200*c[0]/(2*c[0]+c[1]+c[2]) if c[0] else 0.
def read(path):return json.loads(path.read_text())
def audit(root):
 root=Path(root);here=root/'experiments/jev/n8';state=root/'logs/n8'
 manifest=read(state/'evidence-manifest.json')
 for p,want in manifest.items():assert hashlib.sha256((root/p).read_bytes()).hexdigest()==want,p
 requests=read(here/'requests.json');public=read(here/'public.json');summary=read(state/'summary.json');protocol=read(here/'protocol.json');pre=read(state/'preflight.json')
 assert digest(protocol)==pre['protocol_sha256'] and digest(public)==pre['public_sha256'] and pre['gate']=='PASS'
 events=[json.loads(s) for s in (state/'events.jsonl').read_text().splitlines()] if (state/'events.jsonl').exists() else []
 out={'manifest_files_verified':len(manifest),'events':len(events),'all_complete':False,'gold_read_by_audit':False}
 costs={}
 for e in events:
  if e.get('batch_id'):
   if e['batch_id'] in costs:assert costs[e['batch_id']]==e['provider_reported_batch_cost_usd']
   costs[e['batch_id']]=e['provider_reported_batch_cost_usd']
 out['batch_reported_USD']=sum(v for v in costs.values() if v is not None)
 assert abs(out['batch_reported_USD']-summary['new_API_reported_cost_USD'])<1e-12
 if summary['status']!='COMPLETE':
  assert summary['metrics'] is None and not summary['gold_read'];out['status']='BLOCKED_VALIDITY_REPRODUCED';return out
 assert len(events)==len(requests)==56
 by={digest(e['request']['messages']):e for e in events};assert len(by)==56
 allowed={d['id']:{tuple(p) for p in d['candidates']} for d in public};positive=defaultdict(dict);models=defaultdict(int);details=[]
 for r in requests:
  e=by[digest(r['messages'])];body=e['request'];response=e['response'];assert not e.get('error') and not e.get('cache_hit')
  assert body=={'model':protocol['model'],'messages':r['messages'],'reasoning':{'effort':'high'},'max_tokens':32768}
  assert digest(body)==pre['request_hashes'][requests.index(r)]
  assert r['doc'].split('_')[0]!=r['demonstration_id'].split('_')[0]
  assert response['model'] in {'openai/gpt-6-luna','openai/gpt-6-luna:batch','openai/gpt-6-luna-20260922'}
  assert response.get('provider')=='OpenAI' and isinstance(response.get('choices'),list) and len(response['choices'])==1
  models[response['model']]+=1;choice=response['choices'][0];assert choice['finish_reason']=='stop';text=choice['message']['content'];assert isinstance(text,str)
  # An independent JSONDecoder scans every '['; pick last nonempty list of objects.
  arrays=[]
  for i,ch in enumerate(text):
   if ch=='[':
    try:v,end=json.JSONDecoder().raw_decode(text[i:])
    except ValueError:continue
    if isinstance(v,list):arrays.append((i,v))
  assert arrays,'missing final JSON array'
  populated=[(i,a) for i,a in arrays if a];items=(populated or arrays)[-1][1];labels={}
  for item in items:
   assert isinstance(item,dict) and isinstance(item.get('pair'),str) and item['pair'].count(',')==1
   pair=tuple(s.strip() for s in item['pair'].split(','));lab=''.join(item['label'].split()).lower()
   assert pair in allowed[r['doc']] and lab in {'causes','causedby','norel'}
   labels[pair]=lab
  positive[r['doc']][r['pass_index']]={p for p,lab in labels.items() if lab!='norel'}
  details.append({'doc':r['doc'],'pass_index':r['pass_index'],'response_model':response['model'],'finish_reason':choice['finish_reason'],'array_items':len(items),'unique_directed_positive':sum(lab!='norel' for lab in labels.values())})
 predictions=[]
 for d in public:
  assert set(positive[d['id']])==set(range(7))
  for a,b in d['candidates']:predictions.append({'doc':d['id'],'a':a,'b':b,'L3':sum((a,b) in positive[d['id']][i] for i in range(3))/3,'L7':sum((a,b) in positive[d['id']][i] for i in range(7))/7})
 freeze=read(state/'freeze-receipt.json');assert freeze['gold_read'] is False and digest(events)==freeze['events_sha256'] and digest(predictions)==freeze['predictions_sha256']
 assert predictions==read(state/'predictions-before-gold.json')
 # Gold is opened only after independent validity, raw parsing and freeze checks.
 d1rows=read(here/'d1.json');assert digest(d1rows)==freeze['d1_sha256']
 dd={(r['doc'],r['a'],r['b']):r['d1'] for r in d1rows};pp={(r['doc'],r['a'],r['b']):r for r in predictions};assert set(dd)==set(pp)
 fused=[]
 for k,r in pp.items():
  rev=(k[0],k[2],k[1]);row={'doc':k[0],'a':k[1],'b':k[2]}
  for arm,field in [('three','L3'),('seven','L7')]:row[arm]=(r[field]+pp[rev][field]+dd[k]+dd[rev])/4>=.39
  fused.append(row)
 assert fused==read(state/'decisions-before-gold.json') and digest(fused)==freeze['decisions_sha256']
 controls=read(here/'controls.json');out['gold_read_by_audit']=True
 cp={(r['doc'],r['a'],r['b']):r for r in controls};pp={(r['doc'],r['a'],r['b']):r for r in predictions};assert set(cp)==set(pp)
 tc={a:defaultdict(lambda:[0,0,0]) for a in ['three','seven']};decisions=[]
 for k in pp:
  r=cp[k];reverse=(k[0],k[2],k[1]);row={'doc':k[0],'a':k[1],'b':k[2]}
  for arm,field in [('three','L3'),('seven','L7')]:
   pred=(pp[k][field]+pp[reverse][field]+r['d1']+cp[reverse]['d1'])/4>=.39;c=tc[arm][k[0].split('_')[0]];y=r['gold']
   c[0]+=bool(pred and y);c[1]+=bool(pred and not y);c[2]+=bool(not pred and y);row[arm]=pred
  decisions.append(row)
 assert decisions==fused==read(state/'decisions-after-gold.json')
 assert all(r['d1']==dd[k] for k,r in cp.items())
 totals={a:[sum(v[i] for v in tc[a].values()) for i in range(3)] for a in tc};topics=sorted(tc['three']);rng=random.Random(20261029);draws=[]
 for _ in range(10000):
  sample=rng.choices(topics,k=len(topics));cc={a:[sum(tc[a][t][i] for t in sample) for i in range(3)] for a in tc};draws.append(f1(cc['three'])-f1(cc['seven']))
 draws.sort();ci=[draws[250],draws[9749]];decision='NONINFERIOR_DEVELOPMENT' if ci[0]>-1 else ('INFERIOR_DEVELOPMENT' if ci[1]<-1 else 'INCONCLUSIVE')
 expected=summary['metrics'];assert all(totals[a]==expected['systems'][a]['TP_FP_FN'] and f1(totals[a])==expected['systems'][a]['F1'] for a in totals)
 assert ci==expected['topic_bootstrap_CI95'] and decision==expected['decision'] and dict(tc)==expected['per_topic_counts']
 delta=f1(totals['three'])-f1(totals['seven']);assert delta==expected['delta_F1_three_minus_seven']
 loto={t:f1([sum(tc['three'][x][i] for x in topics if x!=t) for i in range(3)])-f1([sum(tc['seven'][x][i] for x in topics if x!=t) for i in range(3)]) for t in topics}
 out.update(status='ALL_RAW_REPLAY_PASS',all_complete=True,models=dict(models),systems=expected['systems'],delta_F1=delta,topic_CI95=ci,decision=decision,per_reply=details,LOTO_delta_F1=loto,training=False,human_annotations=0,limitation='single generation schedule; eight exposed topics; no new independent confirmation')
 return out
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();result=audit(a.root);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=1)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='per_reply'}))
