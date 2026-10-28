"""Independent stdlib replay: python audit_n7.py <extracted-evidence-directory>."""
import collections,hashlib,json,pathlib,random,sys
P=pathlib.Path(sys.argv[1]);read=lambda p:json.loads(p.read_text());sha=lambda b:hashlib.sha256(b).hexdigest()
def f1(c):return 200*c[0]/(2*c[0]+c[1]+c[2]) if c[0] else 0.
m=read(P/'manifest.json')
for n,h in m['files_sha256'].items():assert sha((P/n).read_bytes())==h,n
requests=read(P/'requests-frozen.json');events=[json.loads(x) for x in (P/'events.jsonl').read_text().splitlines()];expected={r['logical_id']:r for r in requests};assert len(expected)==192
assert len({r['body_sha256'] for r in requests})==135
freeze=read(P/'freeze-before-gold.json');assert not freeze['controls_opened'];assert freeze['events_sha256']==sha((P/'events.jsonl').read_bytes());assert freeze['predictions_sha256']==sha((P/'predictions-before-gold.json').read_bytes())
public={d['id']:d for d in read(P/'public.json')};labels={};batches={};slots=collections.Counter()
for e in events:
 k=e['logical_id'];r=expected[k];assert k not in labels and e['request']==r['body'] and not e['cache_hit'] and e['error'] is None
 resp=e['response'];assert resp['model'] in {'openai/gpt-6-luna','openai/gpt-6-luna:batch','openai/gpt-6-luna-20260922'} and resp['provider']=='OpenAI';assert len(resp['choices'])==1 and resp['choices'][0]['finish_reason']=='stop'
 text=resp['choices'][0]['message']['content']
 # N7 gate uses the legacy parser; strict JSON here independently audits normal provider answers.
 items=json.loads(text);assert isinstance(items,list);allowed={tuple(x) for x in public[r['doc']]['pair_list']};labels[k]={}
 for it in items:
  assert isinstance(it,dict) and set(it)=={'pair','label'} and it['label'] in {'causes','causedby','norel'}
  pair=tuple(x.strip() for x in it['pair'].split(','));assert pair in allowed;labels[k][pair]=it['label']
 bid=e['batch_id'];v=e['provider_reported_batch_cost_usd'];assert bid not in batches or batches[bid]==v;batches[bid]=v;slots[bid]+=1
assert set(labels)==set(expected) and len(batches)==64 and set(slots.values())=={3}
for r in requests:
 ev=next(e for e in events if e['logical_id']==r['logical_id']);other=f"s{1-r['schedule']}:{r['arm']}:{r['doc']}:{r['slot']}";oe=next(e for e in events if e['logical_id']==other);assert ev['batch_id']!=oe['batch_id']
assert freeze['decisions_sha256']==sha((P/'decisions-before-gold.json').read_bytes())
frozen_decisions={(r['schedule'],r['arm'],r['doc'],r['a'],r['b']):r for r in read(P/'decisions-before-gold.json')}
controls=read(P/'controls.json');gold={(r['doc'],r['a'],r['b']):r['gold'] for r in controls};d1={(r['doc'],r['a'],r['b']):r['d1'] for r in controls};per={str(s):{a:collections.defaultdict(lambda:[0,0,0]) for a in ('ordinary','contrastive')} for s in (0,1)};stored={(p['schedule'],p['arm'],p['doc'],v['a'],v['b']):v['luna'] for p in read(P/'predictions-before-gold.json') for v in p['scores']}
for s in (0,1):
 for a in ('ordinary','contrastive'):
  sc={}
  for doc,x,y in gold:
   v=sum(labels[f's{s}:{a}:{doc}:{i}'].get((x,y),'norel')!='norel' for i in range(3))/3;assert stored[s,a,doc,x,y]==v;sc[doc,x,y]=v
  for k,y in gold.items():
   rev=k[0],k[2],k[1];assert gold[k]==gold[rev];fused=(sc[k]+sc[rev]+d1[k]+d1[rev])/4;pred=fused>=.39;assert frozen_decisions[s,a,*k]['fused_score']==fused and frozen_decisions[s,a,*k]['positive']==pred;c=per[str(s)][a][k[0].split('_')[0]];c[0]+=int(pred and y);c[1]+=int(pred and not y);c[2]+=int(not pred and y)
topics=sorted(per['0']['ordinary'])
def total(s,a,ts):return [sum(per[str(s)][a][t][i] for t in ts) for i in range(3)]
def delta(s,ts):return f1(total(s,'contrastive',ts))-f1(total(s,'ordinary',ts))
deltas=[delta(s,topics) for s in (0,1)];rng=random.Random(20261028);boot=[]
for _ in range(20000):
 ts=rng.choices(topics,k=len(topics));boot.append(sum(delta(s,ts) for s in (0,1))/2)
boot.sort();ci=[boot[500],boot[19499]];result=read(P/'result.json');assert result['schedule_delta_points']==deltas and result['primary_mean_delta_points']==sum(deltas)/2 and result['topic_bootstrap_CI95_points']==ci
for s in (0,1):
 for a in ('ordinary','contrastive'):assert result['systems'][str(s)][a]['TP_FP_FN']==total(s,a,topics)
cost=sum(batches.values());assert abs(cost-result['cost_known_usd'])<1e-10
out={'audit':'PASS','raw_answers':192,'distinct_batches':64,'distinct_bodies':135,'no_response_cache_reuse':True,'schedule_delta_points':deltas,'mean_delta_points':sum(deltas)/2,'topic_bootstrap_CI95_points':ci,'provider_reported_usd':cost,'files_verified':len(m['files_sha256']),'API_calls_in_audit':0}
print(json.dumps(out,indent=2));(P.parent/'independent-audit.json').write_text(json.dumps(out,indent=2))
