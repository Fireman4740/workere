"""Fixed paired guidance screen; no fitting, no human adjudication."""
import hashlib,json,random
from collections import defaultdict
from pathlib import Path

def preflight(configs):
 from benchmark.configuration import load_suite
 from benchmark.protocols import public_inputs
 from benchmark.data import parse_annotations
 from pipelines.jev.llm import marked_text,frozen_demonstration_bank
 from pipelines.jev.contrastive import select_demonstrations,negative_pairs
 import yaml
 rows=[]
 for c in configs:
  p=c.data['pipeline']['relation_predictor']['params'];client=p['client'];prompt=yaml.safe_load(Path(p['path']).read_text());bank=frozen_demonstration_bank(p['demo_bank_path'],p['repo_id'],p['revision']);assert p['passes']==3
  for public in public_inputs(load_suite(c),'relations_gold'):
   d=public.document;es=public.reference_entities;eligible=[b for b in bank if b['id'].split('_')[0]!=d.id.split('_')[0]]
   demos=select_demonstrations(d,es,bank,3) if p['demo_selection']=='contrastive_structural' else random.Random(f'{c.data["seed"]}::{d.id}').sample(eligible,3)
   for i,demo in enumerate(demos):
    answer=[{'pair':f'{a},{b}','label':lab} for a,lab,b in parse_annotations(demo['annots']) if lab!='norel']
    if p['demo_selection']=='contrastive_structural':answer += [{'pair':f'{x},{y}','label':'norel'} for a,b in negative_pairs(demo) for x,y in [(a,b),(b,a)]]
    messages=[{'role':'system','content':prompt['system']},{'role':'user','content':prompt['user_template'].format(doc_text=demo['text'],pair_lines='',doc_id=demo['id'])},{'role':'assistant','content':json.dumps(answer)},{'role':'user','content':prompt['user_template'].format(doc_text=marked_text(d,es),pair_lines='',doc_id=d.id)}]
    body={'model':client['model'],'messages':messages,'reasoning':{'effort':client['reasoning_effort']},'max_tokens':client['max_output_tokens']};bound=len(json.dumps(body,ensure_ascii=False).encode())+1024
    assert bound+client['max_output_tokens']<=client['max_context_tokens'],(d.id,bound)
    assert bound*client['input_price_per_million']/1e6+client['max_output_tokens']*.5/1e6<=client['max_cost_per_call_usd']
    rows.append({'config':c.data['name'],'doc':d.id,'pass':i,'demo':demo['id'],'input_utf8_bound':bound,'request_sha256':hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()})
 assert len(rows)==72 and len({r['request_sha256'] for r in rows})==72
 result={'gate':'PASS','planned_requests':72,'reserve_USD':1.8,'cap_USD':2,'max_input_utf8_bound':max(r['input_utf8_bound'] for r in rows),'requests':rows};print('N3_PREFLIGHT '+json.dumps(result),flush=True);return result

def f1(c):
 t,p,n=c;return 200*t/(2*t+p+n) if t else 0.
def evaluate(receipt,control_path):
 controls=json.loads(Path(control_path).read_text());scores={'ordinary':{},'contrastive':{}};costs=defaultdict(float);tokens=defaultdict(lambda:defaultdict(int));validity=[]
 for run in receipt['runs']:
  directory=Path(run['run_dir']);arm='contrastive' if 'contrastive' in directory.name else 'ordinary';validity.append({'run_dir':str(directory),'status':run['status']})
  if run['status']!='completed':continue
  for row in map(json.loads,(directory/'predictions.jsonl').read_text().splitlines()):
   assert row['metadata']['passes']==3 and len(row['metadata']['pass_triples'])==3
   for s in row['metadata']['decision_scores']:scores[arm][row['id'],s['source'],s['target']]=s['positive_probability']
  for path in [directory/'decisions.jsonl']:
   if path.exists():
    for e in map(json.loads,path.read_text().splitlines()):
     costs[arm]+=e.get('cost_usd') or 0
     for k,v in (e.get('response',{}).get('usage') or {}).items():
      if isinstance(v,int):tokens[arm][k]+=v
 result={'study':'N3','status':'INCONCLUSIVE','role':'exploratory exposed DEV, no independent confirmation','threshold':.39,'new_API_cost_reported_USD':dict(costs),'token_usage':dict(tokens),'validity':validity}
 keys={(r['doc'],r['a'],r['b']) for r in controls}
 if any(set(s)!=keys for s in scores.values()) or any(v['status']!='completed' for v in validity):return result
 gold={ (r['doc'],r['a'],r['b']):r['gold'] for r in controls};d1={(r['doc'],r['a'],r['b']):r['d1'] for r in controls};topiccounts={arm:defaultdict(lambda:[0,0,0]) for arm in scores};summaries={}
 for arm,score in scores.items():
  for k,y in gold.items():
   rev=(k[0],k[2],k[1]);pred=(score[k]+score[rev]+d1[k]+d1[rev])/4>=.39;c=topiccounts[arm][k[0].split('_')[0]];c[0]+=pred and y;c[1]+=pred and not y;c[2]+=(not pred) and y
  c=[sum(v[i] for v in topiccounts[arm].values()) for i in range(3)];summaries[arm]={'TP_FP_FN':c,'F1':f1(c),'precision':100*c[0]/max(c[0]+c[1],1),'recall':100*c[0]/max(c[0]+c[2],1)}
 topics=sorted(topiccounts['ordinary']);rng=random.Random(20261028);ds=[]
 for _ in range(5000):
  draw=rng.choices(topics,k=len(topics));cs=[[sum(topiccounts[a][t][i] for t in draw) for i in range(3)] for a in ['ordinary','contrastive']];ds.append(f1(cs[1])-f1(cs[0]))
 ds.sort();delta=summaries['contrastive']['F1']-summaries['ordinary']['F1'];ci=[ds[125],ds[4874]];result.update(status='COMPLETE',documents=len({k[0] for k in gold}),topics=len(topics),systems=summaries,delta_F1=delta,topic_bootstrap_CI95=ci,development_candidate=delta>=1 and ci[0]>0,per_topic_counts=topiccounts,limitations=['one fresh three-read schedule per arm','same allocation and ceilings, not equal realized reasoning tokens or invoice','old DEV exposed, no new gold annotations requested']);return result
