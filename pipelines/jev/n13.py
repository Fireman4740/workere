"""Tracked local Qwen representation study. No external model/provider fallback."""
import base64,collections,concurrent.futures,datetime,hashlib,io,json,os,pathlib,random,shlex,sys,threading,time,urllib.error,urllib.parse,urllib.request,zipfile

P=pathlib.Path('experiments/jev/n13')
ARMS=('original','local','table')
read=lambda p:json.loads(pathlib.Path(p).read_text())
canonical=lambda v:json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
sha=lambda b:hashlib.sha256(b).hexdigest()

def save(p,value):
    p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=1,allow_nan=False)+'\n');tmp.replace(p)

def credential(spec):
    name=spec['credential_name']
    if os.environ.get(name):return os.environ[name]
    for line in pathlib.Path(spec['credential_file']).expanduser().read_text().splitlines():
        k,sep,raw=line.strip().removeprefix('export ').partition('=')
        if sep and k.strip()==name:
            values=shlex.split(raw,comments=True)
            if len(values)==1 and values[0]:return values[0]
    raise RuntimeError('local service credential unavailable')

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):return None

class LocalHTTP:
    def __init__(self,key):
        self.key=key;self.opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect())
    def __call__(self,method,url,payload=None,timeout=60):
        parts=urllib.parse.urlparse(url)
        if (parts.scheme,parts.hostname,parts.port)!=('http','127.0.0.1',18020):
            raise ValueError('external inference and nonlocal transport prohibited')
        if parts.path not in ('/v1/models','/tokenize','/v1/tokenize','/openapi.json','/v1/chat/completions'):
            raise ValueError('unapproved local service path')
        req=urllib.request.Request(url,method=method,data=json.dumps(payload).encode() if payload is not None else None,
            headers={'Authorization':'Bearer '+self.key,'Content-Type':'application/json'})
        with self.opener.open(req,timeout=timeout) as response:return json.load(response)

def active_dependency_processes(run):
    # Process CWD identifies the existing tracked job. Never read credentials or /proc/environ.
    proc=pathlib.Path('/proc')
    if not proc.is_dir():raise RuntimeError('Linux execution host required for dependency inspection')
    active=[]
    for p in proc.iterdir():
        if not p.name.isdigit():continue
        try:cwd=os.readlink(p/'cwd')
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        if '/.orx/runs/'+run+'/' in cwd or cwd.endswith('/.orx/runs/'+run):active.append(int(p.name))
    return sorted(active)

def wait_for_dependency(spec,state):
    deadline=time.monotonic()+spec['queue_timeout_hours']*3600
    while True:
        active=active_dependency_processes(spec['wait_for_run'])
        status={'phase':'WAITING_FOR_EXISTING_LOCAL_BASELINE' if active else 'LOCAL_CAPACITY_RELEASED',
                'dependency_run':spec['wait_for_run'],'dependency_processes':active,'new_generation_calls':0,
                'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
        save(state/'queue-status.json',status);print('N13_QUEUE '+json.dumps(status),flush=True)
        if not active:return
        if time.monotonic()>=deadline:raise RuntimeError('queue deadline reached; no generation submitted')
        time.sleep(60)

def preflight():
    spec=read(P/'protocol.json');raw=(P/'requests.jsonl').read_bytes()
    assert sha(raw)==spec['input_sha256']
    rows=[json.loads(l) for l in raw.splitlines()]
    assert len(rows)==len({r['logical_id'] for r in rows})==len({r['custom_id'] for r in rows})==792
    for stage,n in spec['request_counts'].items():
        rs=[r for r in rows if r['stage']==stage];assert len(rs)==n
        docs={d['id']:d for d in read(P/(stage+'-public.json'))}
        for r in rs:
            b=r['body'];assert b['model']==spec['model'] and b['max_tokens']==32768
            assert b['reasoning_effort']=='xhigh' and b['temperature']==0.
            assert b['chat_template_kwargs']=={'enable_thinking':True,'preserve_thinking':True}
            assert sha(canonical(b))==r['body_sha256']
            assert r['demo'].split('_')[0]!=r['doc'].split('_')[0]
            assert r['doc'] in docs and r['arm'] in ARMS
    return spec,rows

def verify_response(response,allowed):
    from eci.parse import answer_array
    assert isinstance(response,dict) and response['model']=='qwen3.8-27b'
    assert len(response['choices'])==1 and response['choices'][0]['finish_reason']=='stop'
    assert isinstance(response.get('id'),str) and response['id']
    items=answer_array(response['choices'][0]['message']['content']);last={}
    for it in items:
        assert isinstance(it,dict) and isinstance(it.get('pair'),str)
        pair=tuple(x.strip() for x in it['pair'].split(','))
        assert pair in allowed and it.get('label') in ('causes','causedby','norel')
        last[pair]=it['label']
    # Binary task: one voice per unordered pair/read, even if both orientations emitted.
    return {tuple(sorted(pair)) for pair,lab in last.items() if lab!='norel'}

def decisions_before_gold(rows,events,public):
    assert len(events)==len(rows) and len({e['logical_id'] for e in events})==len(rows)
    by={e['logical_id']:e for e in events};docs={d['id']:d for d in public};positive={};ids=[]
    for r in rows:
        e=by[r['logical_id']];assert not e['error'] and e['request']==r['body']
        positive[r['logical_id']]=verify_response(e['response'],set(map(tuple,docs[r['doc']]['pair_list'])))
        ids.append(e['response']['id'])
    assert len(ids)==len(set(ids))
    output=[];stage=rows[0]['stage']
    for s in (0,1):
        for arm in ARMS:
            for doc,d in docs.items():
                for a,b in d['pair_list']:
                    if a>b:continue
                    votes=sum(tuple(sorted((a,b))) in positive[f'{stage}:s{s}:{arm}:{doc}:{slot}'] for slot in range(3))
                    output.append({'schedule':s,'arm':arm,'doc':doc,'a':a,'b':b,'votes':votes,'positive':votes>=2})
    return output

def evaluate(decisions,gold_rows,stage):
    gold={(r['doc'],r['a'],r['b']):r['gold'] for r in gold_rows};per={str(s):{a:collections.defaultdict(lambda:[0,0,0]) for a in ARMS} for s in (0,1)}
    assert len(decisions)==len(gold)*6
    for r in decisions:
        y=gold[r['doc'],r['a'],r['b']];pred=r['positive'];c=per[str(r['schedule'])][r['arm']][r['doc'].split('_')[0]]
        c[0]+=int(pred and y);c[1]+=int(pred and not y);c[2]+=int(not pred and y)
    topics=sorted(per['0']['original'])
    def score(s,a,ts):
        c=[sum(per[str(s)][a][t][i] for t in ts) for i in range(3)]
        return {'TP_FP_FN':c,'F1':200*c[0]/(2*c[0]+c[1]+c[2]) if c[0] else 0.}
    systems={str(s):{a:score(s,a,topics) for a in ARMS} for s in (0,1)}
    deltas={a:[systems[str(s)][a]['F1']-systems[str(s)]['original']['F1'] for s in (0,1)] for a in ARMS[1:]}
    ci=None
    if stage=='pilot40':
        rng=random.Random(2026103140);boot={a:[] for a in ARMS[1:]}
        for _ in range(10000):
            draw=rng.choices(topics,k=len(topics))
            for a in ARMS[1:]:boot[a].append(sum(score(s,a,draw)['F1']-score(s,'original',draw)['F1'] for s in (0,1))/2)
        ci={a:[sorted(ds)[125],sorted(ds)[9874]] for a,ds in boot.items()}
    return {'study':'N13','stage':stage,'status':'COMPLETE_DEVELOPMENT_ONLY','systems':systems,'schedule_delta_points':deltas,
            'mean_delta_points':{a:sum(ds)/2 for a,ds in deltas.items()},'paired_topic_CI97_5_family2':ci,
            'per_topic_counts':per,'primary':'Qwen-only unordered2of3majority; not historical Luna x d1 fusion',
            'promotion':False,'new_external_API_usd':0,'human_annotations':0,'training':False}

def export(state,label):
    files={str(p.relative_to(state)):sha(p.read_bytes()) for p in state.rglob('*') if p.is_file() and p.suffix not in ('.tmp','.zip') and p.name!='manifest.json'}
    save(state/'manifest.json',{'run':os.environ.get('ORX_RUN_ID'),'commit':os.environ.get('ORX_COMMIT'),'files_sha256':files})
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
        for name in list(files)+['manifest.json']:z.write(state/name,name)
    raw=buf.getvalue();encoded=base64.b64encode(raw).decode();size=6000
    print('N13_'+label+'_EVIDENCE_BEGIN '+json.dumps({'zip_sha256':sha(raw),'base64_chars':len(encoded),'chunks':(len(encoded)+size-1)//size}),flush=True)
    for i in range(0,len(encoded),size):print('N13_'+label+'_EVIDENCE_CHUNK '+json.dumps({'index':i//size,'data':encoded[i:i+size]}),flush=True)
    print('N13_'+label+'_EVIDENCE_END',flush=True)

def tokenize(http,spec,row):
    body={'model':spec['model'],'messages':row['body']['messages'],'add_generation_prompt':True,
          'chat_template_kwargs':row['body']['chat_template_kwargs']}
    root=spec['base_url'].removesuffix('/v1')
    try:response=http('POST',root+'/tokenize',body)
    except urllib.error.HTTPError as exc:
        if exc.code not in (404,405):raise
        response=http('POST',spec['base_url']+'/tokenize',body)
    n=response.get('count');assert type(n) is int and n>0
    assert n<=spec['max_input_tokens'] and n+spec['max_tokens']<=spec['max_context_tokens']
    return {'logical_id':row['logical_id'],'prompt_tokens_exact_service':n,'tokenizer_response':response}

def stage_run(spec,rows,state,http):
    stage=rows[0]['stage'];target=state/stage;target.mkdir();public=read(P/(stage+'-public.json'))
    # Exact service tokenization for every body before any inference in this stage.
    counts=[]
    for r in rows:
        item=tokenize(http,spec,r);counts.append({k:v for k,v in item.items() if k!='tokenizer_response'})
    save(target/'tokenizer-preflight.json',{'gate':'PASS','requests':len(counts),'counts':counts})
    print('N13_STAGE '+json.dumps({'stage':stage,'token_gate':'PASS','requests':len(rows),'concurrency':spec['concurrency']}),flush=True)
    lock=threading.Lock();events=[]
    def generate(r):
        start=time.monotonic();response=None;error=None
        try:
            response=http('POST',spec['base_url']+'/chat/completions',r['body'],timeout=spec['request_timeout_seconds'])
            doc=next(d for d in public if d['id']==r['doc']);verify_response(response,set(map(tuple,doc['pair_list'])))
            usage=response['usage'];assert 0<usage['prompt_tokens']<=spec['max_input_tokens']
            assert 0<=usage['completion_tokens']<=spec['max_tokens']
            assert usage['prompt_tokens']+usage['completion_tokens']<=spec['max_context_tokens']
        except Exception as exc:error={'type':type(exc).__name__,'message':str(exc)[:500]}
        event={'logical_id':r['logical_id'],'request':r['body'],'response':response,'error':error,'seconds':time.monotonic()-start,'generation_attempts':1}
        with lock:
            events.append(event)
            with (target/'events.jsonl').open('a') as f:f.write(json.dumps(event,ensure_ascii=False)+'\n')
            print('N13_PROGRESS '+json.dumps({'stage':stage,'completed':len(events),'planned':len(rows),'errors':sum(bool(e['error']) for e in events)}),flush=True)
        return event
    mixed=list(rows);random.Random('N13::'+stage).shuffle(mixed)
    with concurrent.futures.ThreadPoolExecutor(max_workers=spec['concurrency']) as pool:list(pool.map(generate,mixed))
    if any(e['error'] for e in events):
        result={'stage':stage,'status':'BLOCKED_VALIDITY','valid':sum(not e['error'] for e in events),'expected':len(rows),'read_gold':False,'new_external_API_usd':0}
        save(target/'result.json',result);print('N13_RESULT '+json.dumps(result),flush=True);return False
    decisions=decisions_before_gold(rows,events,public);save(target/'decisions-before-gold.json',decisions)
    save(target/'freeze-before-gold.json',{'decisions_sha256':sha((target/'decisions-before-gold.json').read_bytes()),'events_sha256':sha((target/'events.jsonl').read_bytes()),'gold_opened':False,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    gold=read(P/(stage+'-gold.json'));save(target/'gold-evaluated.json',gold);result=evaluate(decisions,gold,stage)
    save(target/'result.json',result);print('N13_RESULT '+json.dumps(result),flush=True);return True

def main():
    spec,rows=preflight()
    if '--preflight' in sys.argv:
        print(json.dumps({'gate':'PASS','stage_requests':spec['request_counts'],'API_cost':0,'queued_after':spec['wait_for_run']}));return
    run=os.environ['ORX_RUN_ID'];state=pathlib.Path('~/.local/share/openresearch/workere-n13').expanduser()/run
    assert not state.exists(),'fresh tracked state required, no automatic inference resubmission'
    state.mkdir(parents=True)
    for p in P.glob('*'):
        if p.is_file() and 'gold' not in p.name:(state/p.name).write_bytes(p.read_bytes())
    (state/'n13.py').write_bytes(pathlib.Path(__file__).read_bytes())
    try:
        http=LocalHTTP(credential(spec));models=http('GET',spec['base_url']+'/models')
        row=next(r for r in models['data'] if r['id']==spec['model']);assert row.get('max_model_len',49152)>=49152
        save(state/'served-models.json',models)
        save(state/'tokenizer-probe.json',tokenize(http,spec,rows[0]))
        print('N13_LOCAL_MODEL_READY '+json.dumps({'model':row['id'],'reported_max_model_len':row.get('max_model_len'),'external_API_calls':0}),flush=True)
        wait_for_dependency(spec,state)
        for stage in ('smoke','pilot40'):
            good=stage_run(spec,[r for r in rows if r['stage']==stage],state,http)
            export(state,stage.upper())
            if not good:return
        save(state/'completion.json',{'status':'COMPLETE','generation_calls':792,'API_usd':0,'human_annotations':0})
    except Exception as exc:
        save(state/'controller-error.json',{'type':type(exc).__name__,'message':str(exc)[:1000],'no_external_fallback':True,'no_generation_retry':True})
        print('N13_FAILURE '+json.dumps(read(state/'controller-error.json')),flush=True)
        export(state,'FAILURE');raise

if __name__=='__main__':main()
