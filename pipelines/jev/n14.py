"""Parallel, paired N14 medium factorial. Local inference only; no training."""
import collections
import concurrent.futures
import copy
import datetime
import json
import os
import pathlib
import random
import sys
import threading
import time
from pipelines.jev import n13 as h
from eci.parse import answer_array

P=pathlib.Path('experiments/jev/n14')
CONDITIONS=('M00','M10','M01','M11')
PRINT_LOCK=threading.Lock()

def log(marker,value):
    with PRINT_LOCK:print('N14_'+marker+' '+json.dumps(value),flush=True)

def preflight():
    spec=h.read(P/'protocol.json');raw=(P/'requests.jsonl').read_bytes()
    assert h.sha(raw)==spec['input_sha256']
    rows=[json.loads(x) for x in raw.splitlines()];docs=h.read(P/'public.json')
    assert len(rows)==len({r['logical_id'] for r in rows})==spec['maximum_generation_calls']
    assert len(docs)==4 and 'WSJ910225-0066' in {d['id'] for d in docs}
    bank={d['id']:d for d in h.read(P/'demo-bank.json')['documents']}
    docmap={d['id']:d for d in docs};coverage=collections.Counter()
    for r in rows:
        b=r['body'];assert h.sha(h.canonical(b))==r['body_sha256']
        assert b['reasoning_effort']==b['chat_template_kwargs']['reasoning_effort']=='medium'
        assert b['max_tokens']==spec['max_tokens']==65536 and b['temperature']==0.
        assert b['model']==spec['model'] and r['condition'] in CONDITIONS
        assert r['demo'] not in docmap and bank[r['demo']]['fold']!=int(docmap[r['doc']]['idx'])%10
        pairs={tuple(sorted(p)) for p in r['allowed']};assert pairs
        assert pairs<={tuple(sorted(p)) for p in docmap[r['doc']]['pair_list']}
        if r['bounded']:assert len(pairs)<=32 and len(r['allowed'])==len(pairs)
        for pair in pairs:coverage[r['condition'],r['schedule'],r['slot'],r['arm'],r['doc'],pair]+=1
    for condition in CONDITIONS:
        assert sum(r['condition']==condition for r in rows)==spec['request_counts'][condition]
        for d in docs:
            for pair in {tuple(sorted(p)) for p in d['pair_list']}:
                for s in (0,1):
                    for slot in range(3):
                        for arm in h.ARMS:assert coverage[condition,s,slot,arm,d['id'],pair]==1
    return spec,rows,docs

def validate(response,row):
    if not row['bounded']:return h.verify_response(response,set(map(tuple,row['allowed'])))
    assert response['model']=='qwen3.8-27b' and len(response['choices'])==1
    assert response['choices'][0]['finish_reason']=='stop','truncated completion'
    assert isinstance(response.get('id'),str) and response['id']
    content=response['choices'][0]['message']['content']
    assert isinstance(content,str),'missing final content'
    items=answer_array(content);seen=set();positive=set();allowed=set(map(tuple,row['allowed']))
    for item in items:
        assert isinstance(item,dict) and isinstance(item.get('pair'),str)
        pair=tuple(x.strip() for x in item['pair'].split(','))
        assert pair in allowed and pair not in seen,'unknown, reversed or duplicate requested pair'
        assert item.get('label') in ('causes','causedby','norel')
        seen.add(pair)
        if item['label']!='norel':positive.add(tuple(sorted(pair)))
    assert seen==allowed,'missing requested pair labels'
    return positive

def template_gate(http,spec,state):
    probes={}
    for effort in ('medium','xhigh'):
        row={'logical_id':'effort-'+effort,'body':{'messages':[{'role':'user','content':'Return the JSON array [].'}],
             'chat_template_kwargs':{'enable_thinking':True,'preserve_thinking':True,'reasoning_effort':effort}}}
        probes[effort]=h.tokenize(http,spec,row)['tokenizer_response']
    assert probes['medium'].get('tokens') and probes['xhigh'].get('tokens'),'token IDs required for effort gate'
    assert probes['medium']['tokens']!=probes['xhigh']['tokens'],'medium and xhigh render identically; no inference'
    h.save(state/'effort-tokenizer-probes.json',probes)
    log('EFFORT_GATE',{'status':'PASS','medium_and_xhigh_token_streams_differ':True,'generation_calls':0})

def run_condition(condition,rows,http,spec,state):
    target=state/condition;target.mkdir();h.save(target/'requests.json',rows)
    counts={};events=[];lock=threading.Lock()
    try:
        for row in rows:
            item=h.tokenize(http,spec,row);counts[row['logical_id']]=item['prompt_tokens_exact_service']
        h.save(target/'tokenizer-preflight.json',{'status':'PASS','counts':counts})
        log('CONDITION_READY',{'condition':condition,'requests':len(rows),'concurrency':spec['condition_concurrency']})
        def generate(row):
            start=time.monotonic();response=None;error=None
            try:
                response=http('POST',spec['base_url']+'/chat/completions',row['body'],timeout=spec['request_timeout_seconds'])
                validate(response,row)
                usage=response['usage']
                assert usage['prompt_tokens']==counts[row['logical_id']],'rendered token count differs from actual prompt'
                assert 0<=usage['completion_tokens']<=spec['max_tokens']
                assert usage['prompt_tokens']+usage['completion_tokens']<=spec['max_context_tokens']
            except Exception as exc:error={'type':type(exc).__name__,'message':str(exc)[:500]}
            event={'logical_id':row['logical_id'],'request':row['body'],'response':response,'error':error,
                   'seconds':time.monotonic()-start,'generation_attempts':1}
            with lock:
                events.append(event)
                with (target/'events.jsonl').open('a') as f:f.write(json.dumps(event,ensure_ascii=False)+'\n')
                log('PROGRESS',{'condition':condition,'completed':len(events),'planned':len(rows),'errors':sum(bool(e['error']) for e in events)})
        mixed=list(rows);random.Random('N14::'+condition).shuffle(mixed)
        with concurrent.futures.ThreadPoolExecutor(max_workers=spec['condition_concurrency']) as pool:list(pool.map(generate,mixed))
        return events
    except Exception as exc:
        h.save(target/'condition-error.json',{'type':type(exc).__name__,'message':str(exc)[:500]})
        log('CONDITION_BLOCKED',{'condition':condition,'type':type(exc).__name__,'message':str(exc)[:500]})
        return events

def freeze_decisions(condition,rows,events,docs):
    by={e['logical_id']:e for e in events};assert len(by)==len(events)
    ids=[e['response']['id'] for e in events if e['response'] and e['response'].get('id')]
    assert len(ids)==len(set(ids)),'duplicate response ID'
    complete=[];coverage={}
    for d in docs:
        rs=[r for r in rows if r['doc']==d['id']]
        valid=sum(r['logical_id'] in by and not by[r['logical_id']]['error'] for r in rs)
        coverage[d['id']]={'planned_calls':len(rs),'delivered_calls':sum(r['logical_id'] in by for r in rs),'valid_calls':valid}
        if valid==len(rs):complete.append(d['id'])
    votes=collections.Counter()
    for row in rows:
        if row['doc'] not in complete:continue
        event=by[row['logical_id']];assert event['request']==row['body']
        for pair in validate(event['response'],row):votes[row['schedule'],row['arm'],row['doc'],pair]+=1
    decisions=[]
    for s in (0,1):
        for arm in h.ARMS:
            for d in docs:
                if d['id'] not in complete:continue
                for a,b in sorted({tuple(sorted(p)) for p in d['pair_list']}):
                    v=votes[s,arm,d['id'],(a,b)];assert 0<=v<=3
                    decisions.append({'schedule':s,'arm':arm,'doc':d['id'],'a':a,'b':b,'votes':v,'positive':v>=2})
    return {'condition':condition,'complete_documents':complete,'coverage':coverage,'decisions':decisions}

def finish(rows,events_by,docs,state):
    frozen={}
    for condition in CONDITIONS:
        frozen[condition]=freeze_decisions(condition,[r for r in rows if r['condition']==condition],events_by[condition],docs)
        h.save(state/condition/'decisions-before-gold.json',frozen[condition])
    common=set.intersection(*(set(x['complete_documents']) for x in frozen.values()))
    h.save(state/'freeze-before-gold.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'condition_sha256':{c:h.sha(h.canonical(x)) for c,x in frozen.items()},'common_complete_documents':sorted(common),
           'gold_opened_by_controller':False,'development_exposure':'previous diagnostics already exposed these development documents'})
    gold=h.read(P/'gold.json');h.save(state/'gold-evaluated.json',gold)
    result={'study':'N14-CTB4-2x2','conditions':{},'common_complete_documents':sorted(common),'all4_complete_all_conditions':len(common)==4,
            'new_external_API_usd':0,'human_annotations':0,'training':False,'not_equal_cost':True,'confirmation':False}
    for c,x in frozen.items():
        g=[r for r in gold if r['doc'] in x['complete_documents']]
        score=h.evaluate(x['decisions'],g,'n14') if g else None
        common_dec=[r for r in x['decisions'] if r['doc'] in common]
        common_gold=[r for r in gold if r['doc'] in common]
        paired=h.evaluate(common_dec,common_gold,'n14-common') if common_gold else None
        events=events_by[c]
        usage={k:sum((e['response'] or {}).get('usage',{}).get(k,0) for e in events) for k in ('prompt_tokens','completion_tokens','total_tokens')}
        result['conditions'][c]={'status':'COMPLETE_CTB4' if len(x['complete_documents'])==4 else 'INCOMPLETE_DIAGNOSTIC_ONLY',
                               'complete_documents':x['complete_documents'],'coverage':x['coverage'],'planned_calls':sum(r['condition']==c for r in rows),
                               'actual_calls':len(events),'invalid_calls':sum(bool(e['error']) for e in events),'usage':usage,
                               'score_complete_documents':score,'score_common_complete_subset':paired}
        log('RESULT',{'condition':c,**result['conditions'][c]})
    h.save(state/'results.json',result)
    return result

def main():
    spec,rows,docs=preflight()
    if '--preflight' in sys.argv:log('PREFLIGHT',{'status':'PASS','calls':spec['request_counts'],'external_API_usd':0});return
    state=pathlib.Path(spec['state_root']).expanduser()/os.environ['ORX_RUN_ID']
    assert not state.exists(),'fresh tracked state required'
    state.mkdir(parents=True)
    for name in ('protocol.json','requests.jsonl','public.json','demo-bank.json','demo-public.json'):(state/name).write_bytes((P/name).read_bytes())
    (state/'n14.py').write_bytes(pathlib.Path(__file__).read_bytes())
    try:
        http=h.LocalHTTP(h.credential(spec));model=h.ready_model(http,spec,state)
        log('CONFIG',{'model':model['id'],'reasoning_effort':'medium','input':spec['max_input_tokens'],'output':spec['max_tokens'],
                      'context':model.get('max_model_len'),'max_concurrent':spec['concurrency'],'calls':spec['request_counts']})
        template_gate(http,spec,state)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            tasks={c:pool.submit(run_condition,c,[r for r in rows if r['condition']==c],http,spec,state) for c in CONDITIONS}
            events_by={c:f.result() for c,f in tasks.items()}
        finish(rows,events_by,docs,state)
    except Exception as exc:
        h.save(state/'controller-error.json',{'type':type(exc).__name__,'message':str(exc)[:1000]})
        log('FAILURE',h.read(state/'controller-error.json'));raise
    finally:h.export(state,'N14_FINAL')

if __name__=='__main__':main()
