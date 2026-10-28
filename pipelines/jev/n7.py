"""N7 frozen paired two-schedule replication with pre-gold raw export."""
import base64, collections, concurrent.futures, hashlib, io, json, os, pathlib, random, sys, threading, zipfile
P=pathlib.Path('experiments/jev/n7')
ARMS=('ordinary','contrastive')
def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(b):return hashlib.sha256(b).hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def f1(c):return 200*c[0]/(2*c[0]+c[1]+c[2]) if c[0] else 0.
def prepare():
 import yaml
 from benchmark.schema import Document,Entity
 from benchmark.data import parse_annotations
 from pipelines.jev.llm import marked_text
 from pipelines.jev.contrastive import select_demonstrations,negative_pairs
 docs=read(P/'public.json');protocol=read(P/'protocol.json');bank=read('experiments/jev/round4/demo-bank.json')['documents'];prompt=yaml.safe_load(pathlib.Path('prompts/esl_standard_eci_lean_coref.yaml').read_text());requests=[]
 for schedule,seed in enumerate(protocol['schedule_seeds']):
  for d in docs:
   doc=Document(d['id'],tuple(d['tokens']),tuple(tuple(x) for x in d['sentences']),d['lang']);es=tuple(Entity(m,tuple(sp)) for m,sp in zip(d['mentions'],d['spans']));eligible=[b for b in bank if b['id'].split('_')[0]!=doc.id.split('_')[0]]
   for arm in ARMS:
    demos=select_demonstrations(doc,es,bank,3) if arm=='contrastive' else random.Random(f'{seed}::{doc.id}').sample(eligible,3)
    for slot,demo in enumerate(demos):
     answer=[{'pair':f'{a},{b}','label':lab} for a,lab,b in parse_annotations(demo['annots']) if lab!='norel']
     if arm=='contrastive':answer += [{'pair':f'{x},{y}','label':'norel'} for a,b in negative_pairs(demo) for x,y in [(a,b),(b,a)]]
     messages=[{'role':'system','content':prompt['system']},{'role':'user','content':prompt['user_template'].format(doc_text=demo['text'],pair_lines='',doc_id=demo['id'])},{'role':'assistant','content':json.dumps(answer)},{'role':'user','content':prompt['user_template'].format(doc_text=marked_text(doc,es),pair_lines='',doc_id=doc.id)}]
     body={'model':'openai/gpt-6-luna:batch','messages':messages,'reasoning':{'effort':'high'},'max_tokens':32768};bound=len(json.dumps(body,ensure_ascii=False).encode())+1024
     assert bound+32768<=65536 and bound*.1/1e6+32768*.5/1e6<=.025
     requests.append({'logical_id':f's{schedule}:{arm}:{doc.id}:{slot}','schedule':schedule,'seed':seed,'arm':arm,'doc':doc.id,'slot':slot,'demo':demo['id'],'body':body,'body_sha256':sha(canonical(body)),'input_utf8_bound':bound})
 assert len(requests)==192 and len({r['logical_id'] for r in requests})==192
 return requests

def validate_content(response,allowed):
 from eci.parse import parse_triples,answer_array
 assert response['model'] in {'openai/gpt-6-luna','openai/gpt-6-luna:batch','openai/gpt-6-luna-20260922'} and response.get('provider')=='OpenAI'
 assert len(response['choices'])==1 and response['choices'][0]['finish_reason']=='stop'
 content=response['choices'][0]['message']['content'];items=answer_array(content);triples=parse_triples(content)
 assert len(items)==len(triples),'malformed pair skipped by legacyparser'
 assert all((a,b) in allowed and lab in {'causes','causedby','norel'} for a,lab,b in triples)
 return {(a,b):lab for a,lab,b in triples} # legacy last-label-wins within eachpass

def freeze_predictions(requests,events):
 docs={d['id']:d for d in read(P/'public.json')};by={e['logical_id']:e for e in events};assert len(by)==len(events)==192
 labels={};failures=[]
 for r in requests:
  e=by[r['logical_id']]
  try:
   assert not e['cache_hit'] and e['error'] is None
   assert e['request']==r['body']
   labels[r['logical_id']]=validate_content(e['response'],{tuple(x) for x in docs[r['doc']]['pair_list']})
  except Exception as exc:failures.append({'logical_id':r['logical_id'],'type':type(exc).__name__,'message':str(exc)})
 if failures:return None,failures
 predictions=[]
 for schedule in (0,1):
  for arm in ARMS:
   for doc,d in docs.items():
    passes=[labels[f's{schedule}:{arm}:{doc}:{i}'] for i in range(3)]
    sc={(a,b):sum(l.get((a,b),'norel')!='norel' for l in passes)/3 for a,b in d['pair_list']}
    predictions.append({'schedule':schedule,'arm':arm,'doc':doc,'scores':[{'a':a,'b':b,'luna':v} for (a,b),v in sorted(sc.items())]})
 return predictions,[]

def evaluate(predictions,controls,draws=20000):
 keys={(r['doc'],r['a'],r['b']) for r in controls};gold={(r['doc'],r['a'],r['b']):r['gold'] for r in controls};d1={(r['doc'],r['a'],r['b']):r['d1'] for r in controls};per={str(s):{a:collections.defaultdict(lambda:[0,0,0]) for a in ARMS} for s in (0,1)}
 for schedule in (0,1):
  for arm in ARMS:
   sc={(r['doc'],v['a'],v['b']):v['luna'] for r in predictions if r['schedule']==schedule and r['arm']==arm for v in r['scores']};assert set(sc)==keys
   for k,y in gold.items():
    rev=k[0],k[2],k[1];assert gold[k]==gold[rev];pred=(sc[k]+sc[rev]+d1[k]+d1[rev])/4>=.39;c=per[str(schedule)][arm][k[0].split('_')[0]];c[0]+=int(pred and y);c[1]+=int(pred and not y);c[2]+=int(not pred and y)
 topics=sorted(per['0']['ordinary'])
 def total(s,a,ts):return [sum(per[str(s)][a][t][i] for t in ts) for i in range(3)]
 def delta(s,ts):return f1(total(s,'contrastive',ts))-f1(total(s,'ordinary',ts))
 systems={str(s):{a:{'TP_FP_FN':total(s,a,topics),'F1':f1(total(s,a,topics))} for a in ARMS} for s in (0,1)};deltas=[delta(s,topics) for s in (0,1)];rng=random.Random(20261028);boot=[]
 for _ in range(draws):
  ts=rng.choices(topics,k=len(topics));boot.append(sum(delta(s,ts) for s in (0,1))/2)
 boot.sort();ci=[boot[int(draws*.025)],boot[int(draws*.975)-1]];mean=sum(deltas)/2
 return {'status':'COMPLETE','systems':systems,'schedule_delta_points':deltas,'primary_mean_delta_points':mean,'topic_bootstrap_CI95_points':ci,'continuation_candidate':mean>=1 and ci[0]>0 and min(deltas)>0,'LOTO_mean_delta_points':{t:sum(delta(s,[x for x in topics if x!=t]) for s in (0,1))/2 for t in topics},'per_topic_counts':per,'topics':len(topics),'documents':len({r['doc'] for r in controls}),'ordered_candidates':len(keys),'interpretation':'exploratory exposed development; no production promotion'}

def export(root):
 files={str(p.relative_to(root)):sha(p.read_bytes()) for p in root.rglob('*') if p.is_file() and p.suffix not in {'.sqlite','.lock','.tmp'} and not p.name.endswith(('.sqlite-wal','.sqlite-shm'))};save(root/'manifest.json',{'run':os.environ.get('ORX_RUN_ID'),'commit':os.environ.get('ORX_COMMIT'),'files_sha256':files})
 buffer=io.BytesIO()
 with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
  for p in root.rglob('*'):
   if p.is_file() and (str(p.relative_to(root)) in files or p.name=='manifest.json'):z.write(p,str(p.relative_to(root)))
 enc=base64.b64encode(buffer.getvalue()).decode();size=6000
 print('N7_EVIDENCE_BEGIN '+json.dumps({'zip_sha256':sha(buffer.getvalue()),'base64_chars':len(enc),'chunks':(len(enc)+size-1)//size}),flush=True)
 for i in range(0,len(enc),size):print('N7_EVIDENCE_CHUNK '+json.dumps({'index':i//size,'data':enc[i:i+size]}),flush=True)
 print('N7_EVIDENCE_END',flush=True)

def main():
 requests=prepare();pre={'gate':'PASS','requests':len(requests),'distinct_request_bodies':len({r['body_sha256'] for r in requests}),'logical_slots':192,'full_reservation_usd':4.8,'hard_cap_usd':5,'max_input_bound':max(r['input_utf8_bound'] for r in requests)};print('N7_PREFLIGHT '+json.dumps(pre),flush=True)
 if '--preflight' in sys.argv:save(P/'preflight-frozen.json',pre);save(P/'requests-frozen.json',requests);return
 run=os.environ['ORX_RUN_ID'];root=pathlib.Path('logs/n7')/run;assert not root.exists(),'no automatic runresume or paidretry';root.mkdir(parents=True)
 for name in ('protocol.json','public.json','selection.json','preflight-frozen.json','requests-frozen.json','d1-public.json'):save(root/name,read(P/name))
 (root/'audit_n7.py').write_bytes((P/'audit_n7.py').read_bytes())
 (root/'controller.py').write_bytes(pathlib.Path(__file__).read_bytes())
 save(root/'source-identities.json',{'prompt_sha256':sha(pathlib.Path('prompts/esl_standard_eci_lean_coref.yaml').read_bytes()),'demo_bank_sha256':sha(pathlib.Path('experiments/jev/round4/demo-bank.json').read_bytes()),'controls_sha256':sha((P/'controls.json').read_bytes()),'public_sha256':sha((P/'public.json').read_bytes()),'ORX_RUN_ID':run,'ORX_COMMIT':os.environ.get('ORX_COMMIT')})
 assert canonical(requests)==canonical(read(P/'requests-frozen.json'))
 # Reserve all192 calls once before firstPOST. Perbatch clients cannot add calls.
 from pipelines.jev.spend import SharedBudget
 budget=SharedBudget(path=str(root/'campaign-budget.sqlite'),scope='N7-'+run,limit_usd=5,initial_spend_usd=0);ticket=budget.reserve(4.8);save(root/'reservation-before-calls.json',budget.status());events=[];lock=threading.Lock();groups=collections.defaultdict(list)
 for r in requests:groups[r['logical_id'].rsplit(':',1)[0]].append(r)
 def generate(item):
  gid,rows=item
  from pipelines.jev.batch import BatchChatClient
  client=BatchChatClient(model='openai/gpt-6-luna:batch',model_version='gpt-6-luna-20260922',mode='live',endpoint='https://openrouter.ai/api/v1/batches',cache_path=str(root/'cache'/gid.replace(':','_')/'outputs.sqlite'),retries=0,max_calls=3,budget_usd=.08,max_cost_per_call_usd=.025,input_price_per_million=.1,max_context_tokens=65536,max_output_tokens=32768,reasoning_effort='high',timeout=600,requests_per_minute=6000)
  def callback(r):
   def record(e):
    e={**e,'logical_id':r['logical_id'],'schedule':r['schedule'],'arm':r['arm'],'doc':r['doc'],'slot':r['slot'],'demo':r['demo']}
    with lock:
     events.append(e)
     with (root/'events.jsonl').open('a') as f:f.write(json.dumps(e,ensure_ascii=False)+'\n')
   return record
  try:client.decide_many([r['body']['messages'] for r in rows],[callback(r) for r in rows]);return {'group':gid,'status':'completed'}
  except Exception as exc:return {'group':gid,'status':'failed','type':type(exc).__name__,'message':str(exc)[:1000]}
  finally:client.close()
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:statuses=list(pool.map(generate,groups.items()))
  save(root/'group-status.json',statuses);batches={e['batch_id']:e['provider_reported_batch_cost_usd'] for e in events if e.get('batch_id')};known=sum(v for v in batches.values() if v is not None);unknown=192-len(events)+sum(e.get('provider_reported_batch_cost_usd') is None for e in events);upper=known+unknown*.025
  budget.settle(ticket,upper);save(root/'costs.json',{'known_provider_reported_usd':known,'conservative_usd':upper,'unaccounted_call_slots':unknown,'batch_totals':batches,'perrequest_costs':'allocated equalshare within eachbatch, not measured perrequest','budget':budget.status()})
  predictions,failures=freeze_predictions(requests,events) if len(events)==192 else (None,[{'missing_events':192-len(events)}]);save(root/'validity.json',{'failures':failures,'answers':len(events)})
  if predictions is None:result={'status':'BLOCKED_VALIDITY','primary_mean_delta_points':None,'failures':failures,'read_gold':False}
  else:
   save(root/'predictions-before-gold.json',predictions)
   d1={(r['doc'],r['a'],r['b']):r['d1'] for r in read(P/'d1-public.json')};sc={(r['schedule'],r['arm'],r['doc'],v['a'],v['b']):v['luna'] for r in predictions for v in r['scores']};decisions=[]
   for (schedule,arm,doc,a,b),v in sorted(sc.items()):
    fused=(v+sc[schedule,arm,doc,b,a]+d1[doc,a,b]+d1[doc,b,a])/4;decisions.append({'schedule':schedule,'arm':arm,'doc':doc,'a':a,'b':b,'fused_score':fused,'positive':fused>=.39})
   save(root/'decisions-before-gold.json',decisions);save(root/'freeze-before-gold.json',{'events_sha256':sha((root/'events.jsonl').read_bytes()),'predictions_sha256':sha((root/'predictions-before-gold.json').read_bytes()),'decisions_sha256':sha((root/'decisions-before-gold.json').read_bytes()),'controls_opened':False})
   controls=read(P/'controls.json');assert all(r['d1']==d1[r['doc'],r['a'],r['b']] for r in controls);save(root/'controls.json',controls);result=evaluate(predictions,controls)
  result.update(study='N7',cost_known_usd=known,cost_conservative_usd=upper,planned_request_slots=192,recorded_response_events=len(events),human_annotations=0,training=False);save(root/'result.json',result);print('N7_RESULT '+json.dumps(result),flush=True)
 finally:export(root)
if __name__=='__main__':main()
