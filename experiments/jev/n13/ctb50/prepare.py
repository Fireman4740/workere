import collections,hashlib,json,random,sys
from pathlib import Path
import yaml
B=Path('/Users/mathiscarlesso/.local/share/openresearch')
W=B/'worktrees/75e891b6-a1c1-41a2-a06f-99092098f31e/parallel_n13';sys.path.insert(0,str(W))
from pipelines.jev.n12 import views
H=B/'worktrees/75e891b6-a1c1-41a2-a06f-99092098f31e/chat_dde16780-d6aa-4731-afeb-737127fd99e3'
source=H/'experiments/jev/haiku/data/ctb-suite.json';suite=json.loads(source.read_text());allrows=suite['documents'];by={d['id']:d for d in allrows}
log=B/'run-logs/b41468ee-d81e-4d4c-9df7-1f51309dbde9.5f61e1bf-f986-4acd-8537-99d3884e5a25.log'
plans=[json.loads(l.split(' ',1)[1]) for l in log.read_text().splitlines() if l.startswith('QWEN_PLAN ')]
bankids=plans[0]['r7_ctb']['demo_pool'];assert len(bankids)==20 and set(bankids)<=set(by)
# Selection uses only document IDs, original idx fold and candidate count. Gold accessed later for export/examples.
folds=collections.defaultdict(list)
for d in allrows:
 if d['id'] not in bankids and d['pair_list']:folds[int(d['idx'])%10].append(d)
selected=[];selection={}
for fold,ds in sorted(folds.items()):
 ranked=sorted(ds,key=lambda d:(hashlib.sha256(('N13-CTB50-20261102::'+d['id']).encode()).hexdigest(),d['id']))
 assert len(ranked)>=5;selected+=ranked[:5];selection[str(fold)]={'eligible':len(ranked),'selected':[d['id'] for d in ranked[:5]]}
assert len(selected)==50
ranked=sorted(selected,key=lambda d:(len(d['text'].encode()),d['id']))
smokeids=[ranked[i]['id'] for i in [0,16,33,49]]
fields=('id','idx','text','lang','mentions','spans','sentences','sentence_of','pair_list','tokens','fold')
pub=[{k:d[k] for k in fields} for d in selected]
P=W/'experiments/jev/n13/ctb50';P.mkdir(exist_ok=True)
def save(name,value):(P/name).write_text(json.dumps(value,ensure_ascii=False,indent=1)+'\n')
save('selection.json',{'rule':'exclude fixed20baseline demo pool and zero-candidate docs;5peroriginal_idx_mod10fold, rank SHA256(seed::id); labels/errors unused','seed':'N13-CTB50-20261102','folds':selection,'demo_ids':bankids,'smoke_ids':smokeids,'smoke_rule':'target public UTF8length ranks0,16,33,49 of50; smoke is subset not additional requests'})
save('all-public.json',pub)
prompt=yaml.safe_load((H/'prompts/causal_timebank_syntax.yaml').read_text());save('prompt.json',prompt)
bank=[]
for i in bankids:
 d=by[i];bank.append({'id':i,'fold':int(d['idx'])%10,'text':d['text'],'answer':[{'pair':a+','+b,'label':lab} for a,lab,b in d['gold'] if lab!='norel']})
save('demo-bank.json',{'source':'same fixed20 CTB pool as current local baseline','documents':bank})
def sentence_view(d):
 starts,ends={},{}
 for mid,sp in zip(d['mentions'],d['spans']):
  for i in sp:
   if i-1 not in sp:starts.setdefault(i,[]).append(mid)
   if i+1 not in sp:ends.setdefault(i,[]).append(mid)
 marked=[''.join('<'+mid+' ' for mid in starts.get(i,[]))+t+'>'*len(ends.get(i,[])) for i,t in enumerate(d['tokens'])]
 mentiontokens={i for sp in d['spans'] for i in sp};parts=[];kept=set()
 for sid,(a,b) in enumerate(d['sentences']):
  if any(i in mentiontokens for i in range(a,b)):
   parts.append('[Sentence '+str(sid)+']\n'+' '.join(marked[a:b]));kept.update(range(a,b))
 assert mentiontokens<=kept
 return '\n\n'.join(parts),{'doc':d['id'],'total_sentences':len(d['sentences']),'kept_sentences':len(parts),'kept_tokens':len(kept),'total_tokens':len(marked),'all_mentions_retained':True,'local_equals_original':False,'view':'individually delimited sentences containing mentions; original order/IDs; CTB intra-sentence scope'}

rows=[];viewinfo=[]
for d in pub:
 texts,info=views(d);texts['local'],info=sentence_view(d);viewinfo.append(info);eligible=[b for b in bank if b['fold']!=int(d['idx'])%10]
 for schedule,seed in enumerate([202611020,202611021]):
  demos=random.Random(f'{seed}::{d["id"]}').sample(eligible,3)
  for slot,demo in enumerate(demos):
   prefix=[{'role':'system','content':prompt['system']},{'role':'user','content':prompt['user_template'].format(doc_text=demo['text'])},{'role':'assistant','content':json.dumps(demo['answer'])}]
   for arm in ('original','local','table'):
    body={'model':'qwen3.8-27b','messages':prefix+[{'role':'user','content':prompt['user_template'].format(doc_text=texts[arm])}],
      'reasoning_effort':'xhigh','temperature':0.,'top_p':.95,'top_k':20,'min_p':0.,'presence_penalty':0.,'repetition_penalty':1.,'chat_template_kwargs':{'enable_thinking':True,'preserve_thinking':True},'max_tokens':32768,'stream':False}
    logical=f'ctb50:s{schedule}:{arm}:{d["id"]}:{slot}'
    rows.append({'logical_id':logical,'custom_id':'n13ctb_'+hashlib.sha256(logical.encode()).hexdigest(),'stage':'smoke' if d['id'] in smokeids else 'main','schedule':schedule,'seed':seed,'slot':slot,'arm':arm,'doc':d['id'],'demo':demo['id'],'body':body,'body_sha256':hashlib.sha256(json.dumps(body,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()})
assert len(rows)==900 and len({r['logical_id'] for r in rows})==900
(P/'requests.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in rows))
for stage in ['smoke','main','ctb50']:
 docs=[d for d in pub if stage=='ctb50' or (d['id'] in smokeids)==(stage=='smoke')]
 save(stage+'-public.json',docs)
 gold=[]
 for d in docs:
  pos={tuple(sorted((a,b))) for a,lab,b in by[d['id']]['gold'] if lab!='norel'}
  for a,b in d['pair_list']:
   assert d['sentence_of'][a]==d['sentence_of'][b]
   if a<b:gold.append({'doc':d['id'],'a':a,'b':b,'gold':(a,b) in pos})
 save(stage+'-gold.json',gold)
save('view-statistics.json',viewinfo)
protocol={'study':'N13-CTB50','user_authorization':'2026-11-02 relance N13 sur50items de CTB','model':'qwen3.8-27b','base_url':'http://127.0.0.1:18020/v1','credential_file':'~/qwen38-27b-rtx3090/.env','credential_name':'VLLM_API_KEY',
 'wait_for_run':None,'dataset':'CausalTimeBank-standard-eci-workere','revision':'1bba25094d1dda15f172c1f17bb88c26fb3d767d','documents':50,'folds':10,'candidates':'original CTB intra-sentence pair_list only','stages':['smoke','main'],'request_counts':{'smoke':72,'main':828},'maximum_generation_calls':900,
 'max_input_tokens':16384,'max_tokens':32768,'max_context_tokens':49152,'concurrency':4,'request_timeout_seconds':86400,'api_ready_timeout_seconds':1800,'readiness_poll_seconds':15,'state_root':'~/.local/share/openresearch/workere-n13-ctb50',
 'local_view':'CTB-specific sentence-isolated presentation: each original sentence containing mentions is a separate delimited block, original order/IDs and all mentions retained; no invented semantic links',
 'stages_policy':'4metadata-selected docs first; if all72valid, remaining46; final50 combines raw independent replies with no extra requests; no quality-based selection or generation retries',
 'active_baseline':'b41468ee-d81e-4d4c-9df7-1f51309dbde9 preserved; no service restart or source changes to that run; limited4newinflight',
 'primary_score':'Qwen-only unordered2of3majority per schedule, all original candidate pairs; direction reversals deduplicated within generation',
 'demos':'same positive-answer-only demonstration for all3arms in paired slot; same original idx%10 fold excluded, global target/demo disjoint; no CoT synthesis',
 'comparison_scope':'paired representation study, not a replication of historical7pass generatedCoT baseline; no calibration',
 'uncertainty':'paired document bootstrap10000draws,2contrast Bonferroni97.5percentintervals; no invented CTB topic groups',
 'training':False,'human_annotations':0,'external_API_cost_usd':0,'electricity_and_hardware_cost':'not metered','no_external_fallback':True,'generation_retries':0,'input_sha256':hashlib.sha256((P/'requests.jsonl').read_bytes()).hexdigest(),'source_suite_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
 'model_identity_limit':'declared served model and limits only; no proof immutable weights or independent stochastic reads; temp0 retained',
 'exposure':'existing exposed dataset, not blind confirmation; all gold confined to frozen examples and evaluation exports'}
save('protocol.json',protocol)
print(json.dumps({'requests':len(rows),'stage_counts':collections.Counter(r['stage'] for r in rows),'target_documents':len(pub),'ordered_candidates':sum(len(d['pair_list']) for d in pub),'views_local_same_as_original':sum(v['local_equals_original'] for v in viewinfo)}))
