"""Fresh nested prefix replication; all generation is tracked and bounded."""
from __future__ import annotations
import base64,hashlib,io,json,os,random,sys,traceback,urllib.request,zipfile
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
STATE=ROOT/'logs/n8'

def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def save(path,v):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(v,ensure_ascii=False,indent=1,allow_nan=False)+'\n')
def load(path):return json.loads(path.read_text())
def f1(c):return 200*c[0]/(2*c[0]+c[1]+c[2]) if c[0] else 0.

def positive_pairs(response,allowed):
 from eci.parse import answer_array,parse_triples
 from benchmark.labels import normalize
 if response.get('model') not in {'openai/gpt-6-luna','openai/gpt-6-luna:batch','openai/gpt-6-luna-20260922'}:raise ValueError('unexpected served model')
 if response.get('provider')!='OpenAI' or not isinstance(response.get('choices'),list) or len(response['choices'])!=1:raise ValueError('unexpected provider or choice count')
 choice=response['choices'][0]
 if choice.get('finish_reason')!='stop':raise ValueError('incomplete generation')
 content=choice['message']['content'];items=answer_array(content)
 for item in items:
  if not isinstance(item,dict) or not isinstance(item.get('pair'),str) or item['pair'].count(',')!=1:raise ValueError('malformed pair')
  a,b=(s.strip() for s in item['pair'].split(','))
  if (a,b) not in allowed or not isinstance(item.get('label'),str) or normalize(item['label']) not in {'causes','causedby','norel'}:raise ValueError('unknown ID, loop, candidate or label')
 last={(a,b):lab for a,lab,b in parse_triples(content)}
 return {pair for pair,lab in last.items() if lab!='norel'}

def predictions_from_events(requests,events,public):
 if len(requests)!=56 or len(events)!=56:raise ValueError('not56 complete replies')
 by={digest(e['request']['messages']):e for e in events}
 if len(by)!=56:raise ValueError('duplicate response request')
 allowed={d['id']:{tuple(pair) for pair in d['candidates']} for d in public};passes=defaultdict(dict)
 for r in requests:
  e=by[digest(r['messages'])]
  if e.get('error') or e.get('cache_hit') or not e.get('batch_api'):raise ValueError('error or historical cache used')
  body=e['request']
  if body.get('max_tokens')!=32768 or body.get('reasoning')!={'effort':'high'} or body.get('model')!='openai/gpt-6-luna:batch':raise ValueError('request mismatch')
  passes[r['doc']][r['pass_index']]=positive_pairs(e['response'],allowed[r['doc']])
 output=[]
 for d in public:
  ps=passes[d['id']]
  if set(ps)!=set(range(7)):raise ValueError('wrong slot coverage')
  for a,b in d['candidates']:
   output.append({'doc':d['id'],'a':a,'b':b,'L3':sum((a,b) in ps[i] for i in range(3))/3,'L7':sum((a,b) in ps[i] for i in range(7))/7})
 return output

def fused_before_gold(predictions,d1rows):
 dd={(r['doc'],r['a'],r['b']):r['d1'] for r in d1rows};pp={(r['doc'],r['a'],r['b']):r for r in predictions}
 assert len(dd)==len(d1rows) and set(dd)==set(pp)
 rows=[]
 for k,r in pp.items():
  rev=(k[0],k[2],k[1]);row={'doc':k[0],'a':k[1],'b':k[2]}
  for arm,field in [('three','L3'),('seven','L7')]:row[arm]=(r[field]+pp[rev][field]+dd[k]+dd[rev])/4>=.39
  rows.append(row)
 return rows

def evaluate(predictions,controls):
 cp={(r['doc'],r['a'],r['b']):r for r in controls};pp={(r['doc'],r['a'],r['b']):r for r in predictions}
 assert len(cp)==len(controls) and set(cp)==set(pp)
 tc={arm:defaultdict(lambda:[0,0,0]) for arm in ['three','seven']};decisions=[]
 for k in pp:
  r=cp[k];rev=(k[0],k[2],k[1]);y=r['gold'];row={'doc':k[0],'a':k[1],'b':k[2]}
  for arm,field in [('three','L3'),('seven','L7')]:
   pred=(pp[k][field]+pp[rev][field]+r['d1']+cp[rev]['d1'])/4>=.39
   c=tc[arm][k[0].split('_')[0]];c[0]+=bool(pred and y);c[1]+=bool(pred and not y);c[2]+=bool(not pred and y);row[arm]=pred
  decisions.append(row)
 totals={a:[sum(v[i] for v in tc[a].values()) for i in range(3)] for a in tc};topics=sorted(tc['three']);rng=random.Random(20261029);ds=[]
 for _ in range(10000):
  draw=rng.choices(topics,k=len(topics));c={a:[sum(tc[a][t][i] for t in draw) for i in range(3)] for a in tc};ds.append(f1(c['three'])-f1(c['seven']))
 ds.sort();ci=[ds[250],ds[9749]];decision='NONINFERIOR_DEVELOPMENT' if ci[0]>-1 else ('INFERIOR_DEVELOPMENT' if ci[1]<-1 else 'INCONCLUSIVE')
 return {'status':'COMPLETE','decision':decision,'systems':{a:{'TP_FP_FN':c,'F1':f1(c)} for a,c in totals.items()},'delta_F1_three_minus_seven':f1(totals['three'])-f1(totals['seven']),'topic_bootstrap_CI95':ci,'per_topic_counts':tc,'documents':len({r['doc'] for r in predictions}),'topics':len(topics),'ordered_candidates':len(predictions),'Luna_call_saving_fraction':4/7},decisions

def preflight():
 requests=load(HERE/'requests.json');protocol=load(HERE/'protocol.json');public=load(HERE/'public.json')
 assert len(requests)==56 and len(public)==8 and len({r['doc'].split('_')[0] for r in requests})==8
 assert len({digest(r['messages']) for r in requests})==56
 assert all(r['doc'].split('_')[0]!=r['demonstration_id'].split('_')[0] for r in requests)
 assert all(sorted(r['pass_index'] for r in requests if r['doc']==d['id'])==list(range(7)) for d in public)
 assert not(set(protocol['document_ids']) & set(protocol['selection']['excluded_N7']))
 bodies=[{'model':protocol['model'],'messages':r['messages'],'reasoning':{'effort':'high'},'max_tokens':32768} for r in requests]
 bounds=[len(json.dumps(b,ensure_ascii=False).encode())+1024 for b in bodies]
 assert all(b+32768<=65536 and b*.1/1e6+32768*.5/1e6<=.025 for b in bounds)
 assert 56*.025<=1.5
 result={'gate':'PASS','requests':56,'reservation_USD':56*.025,'cap_USD':1.5,'max_input_utf8_bound':max(bounds),'request_hashes':[digest(b) for b in bodies],'protocol_sha256':digest(protocol),'public_sha256':digest(public),'gold_read':False}
 save(STATE/'preflight.json',result);print('N8_PREFLIGHT '+json.dumps(result),flush=True)
 return requests,protocol,public

def catalog():
 with urllib.request.urlopen('https://openrouter.ai/api/v1/models',timeout=60) as response:data=json.load(response)
 rows=[r for r in data['data'] if r['id']=='openai/gpt-6-luna']
 if len(rows)!=1:raise ValueError('canonical model not in catalog')
 row=rows[0];save(STATE/'catalog-model.json',row)
 if row.get('canonical_slug')!='openai/gpt-6-luna-20260922':raise ValueError('canonical model catalog changed')
 prices=row.get('pricing',{})
 if float(prices['prompt'])>1e-7 or float(prices['completion'])>5e-7:raise ValueError('price exceeds frozen conservative reservation')
 return row

def export():
 paths=list(HERE.glob('*.json'))+list(HERE.glob('*.py'))+[ROOT/'experiments/jev/run.sh']+[p for p in STATE.rglob('*') if p.is_file() and '.sqlite' not in p.name and p.suffix not in {'.db','.lock','.tmp'}]
 # Durable raw batch receipts are inside STATE; no credential or headers included.
 manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths};save(STATE/'evidence-manifest.json',manifest);paths.append(STATE/'evidence-manifest.json')
 buffer=io.BytesIO()
 with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
  for p in paths:z.write(p,str(p.relative_to(ROOT)))
 data=buffer.getvalue();encoded=base64.b64encode(data).decode();chunk=48000
 for i in range(0,len(encoded),chunk):print('N8_EVIDENCE_CHUNK '+json.dumps({'index':i//chunk,'data':encoded[i:i+chunk]}),flush=True)
 print('N8_EVIDENCE_END '+json.dumps({'chunks':(len(encoded)+chunk-1)//chunk,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}),flush=True)

def financial_summary(events):
 by_batch={e['batch_id']:e['provider_reported_batch_cost_usd'] for e in events if e.get('batch_id')}
 totals={k:0 for k in ['prompt_tokens','completion_tokens','total_tokens']}
 for e in events:
  response=e.get('response') if isinstance(e.get('response'),dict) else {}
  usage=response.get('usage') if isinstance(response.get('usage'),dict) else {}
  for k in totals:
   if type(usage.get(k)) in (int,float):totals[k]+=usage[k]
 return {'replies_recorded':len(events),'new_API_reported_cost_USD':sum(v for v in by_batch.values() if v is not None),'batch_costs':by_batch,'usage_totals':totals}

def main():
 if not os.environ.get('ORX_RUN_ID'):raise RuntimeError('tracked run only')
 STATE.mkdir(parents=True,exist_ok=True);events=[];client=None
 summary={'study':'N8','status':'BLOCKED_VALIDITY','gold_read':False,'human_annotations':0,'training':False,'metrics':None}
 try:
  requests,protocol,public=preflight();catalog()
  from pipelines.jev.batch import BatchChatClient
  client=BatchChatClient(model='openai/gpt-6-luna:batch',model_version='gpt-6-luna-20260922',provider='openrouter',endpoint='https://openrouter.ai/api/v1/batches',mode='live',reasoning_effort='high',max_output_tokens=32768,max_context_tokens=65536,cache_path=str(STATE/'cache.sqlite'),max_calls=56,budget_usd=1.5,max_cost_per_call_usd=.025,input_price_per_million=.1,retries=0,timeout=600,requests_per_minute=6000,retry_max_wait_seconds=3600,shared_budget={'path':str(STATE/'budget.sqlite'),'scope':'n8-20261028-cap1.5','limit_usd':1.5,'initial_spend_usd':0})
  def record(e):
   events.append(e)
   with (STATE/'events.jsonl').open('a') as f:f.write(json.dumps(e,ensure_ascii=False,allow_nan=False)+'\n')
  client.decide_many([r['messages'] for r in requests],[record]*56)
  predictions=predictions_from_events(requests,events,public);save(STATE/'predictions-before-gold.json',predictions)
  d1rows=load(HERE/'d1.json');fused=fused_before_gold(predictions,d1rows);save(STATE/'decisions-before-gold.json',fused)
  save(STATE/'freeze-receipt.json',{'decisions_sha256':digest(fused),'d1_sha256':digest(d1rows),'utc':datetime.now(timezone.utc).isoformat(),'gold_read':False,'predictions_sha256':digest(predictions),'events_sha256':digest(events),'ORX_RUN_ID':os.environ['ORX_RUN_ID'],'commit':os.environ.get('ORX_COMMIT')})
  controls=load(HERE/'controls.json');metrics,decisions=evaluate(predictions,controls);assert decisions==fused;save(STATE/'decisions-after-gold.json',decisions)
  summary.update(status='COMPLETE',gold_read=True,metrics=metrics)
 except Exception as exc:
  summary['error']={'type':type(exc).__name__,'message':str(exc)};traceback.print_exc()
 finally:
  if client:
   summary['budget']=client.shared_budget.status();client.close()
  summary.update(financial_summary(events))
  save(STATE/'summary.json',summary);print('N8_RESULT '+json.dumps(summary),flush=True);export()

if __name__=='__main__':main()
