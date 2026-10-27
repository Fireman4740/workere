"""Tracked M1 controller. Two fixed batches; no paid retries; no gold before freeze."""
import base64
from collections import Counter
import fcntl
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.error

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT))
from prepare import save,sha
from preflight import audit
from eci_m0_pilot.protocol import decode_response,json_text,coverage
from eci_m0_pilot.m0 import pair,as_pairs
from eci_m0_pilot.run_pilot import catalog_gate,http,ENDPOINT

def emit(name,obj):print(name+' '+json_text(obj),flush=True)
def export(dest,names=None):
    files={p.name:p.read_text() for p in sorted(dest.iterdir()) if p.is_file() and (names is None or p.name in names)}
    raw=gzip.compress(json_text(files).encode(),mtime=0);encoded=base64.b64encode(raw).decode();h=sha(raw)
    for i in range(0,len(encoded),12000):emit('M1_EXPORT_CHUNK',dict(sha256=h,index=i//12000,data=encoded[i:i+12000]))
    emit('M1_EXPORT_END',dict(sha256=h,chunks=math.ceil(len(encoded)/12000),files=sorted(files)))

def paid_batch(rows,name,state,protocol):
    cfg=protocol['settings'];payload=dict(endpoint='/v1/chat/completions',model=cfg['model'],completion_window='24h',
        requests=[dict(custom_id=r['custom_id'],body=r['body']) for r in rows])
    h=sha(json_text(payload).encode());receipt=state/(name+'-'+h+'.json')
    # State directory is shared by all resumes of this exact frozen protocol.
    with (state/(name+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if receipt.exists():
            record=json.loads(receipt.read_text())
            if not record.get('batch',{}).get('id'):raise RuntimeError('uncertain prior POST; never duplicate submission')
            emit('M1_BATCH_RESUME',dict(phase=name,batch_id=record['batch']['id']))
        else:
            record=dict(state='SUBMITTING',phase=name,payload_sha256=h,logical_requests=len(rows),
                payload=payload,first_run_id=os.environ.get('ORX_RUN_ID'),post_attempts=1,started_unix=time.time(),get_attempts=0,get_errors=0)
            save(receipt,record) # durable before the single paid POST
            try:
                batch=http('POST',ENDPOINT,payload)
                if not isinstance(batch.get('id'),str):raise RuntimeError('missing batch ID')
            except BaseException:
                emit('M1_SUBMISSION_UNCERTAIN',dict(phase=name,receipt=str(receipt),no_retry=True));raise
            record.update(state='ACCEPTED',batch=batch);save(receipt,record)
            emit('M1_BATCH_SUBMITTED',dict(phase=name,batch_id=batch['id'],logical_requests=len(rows),receipt=str(receipt)))
    deadline=time.monotonic()+cfg['poll_deadline_seconds']
    while record['batch'].get('status') not in {'completed','failed','expired','cancelled'}:
        if time.monotonic()>deadline:raise RuntimeError('poll deadline; retain batch ID; retrieval only on resume')
        emit('M1_BATCH_PROGRESS',dict(phase=name,**{k:record['batch'].get(k) for k in ('id','status','request_counts')}))
        time.sleep(30);record['get_attempts']+=1
        try:record['batch']=http('GET',ENDPOINT+'/'+record['batch']['id'])
        except (RuntimeError,urllib.error.URLError,TimeoutError):
            record['get_errors']+=1;save(receipt,record);continue
        save(receipt,record) # save raw batch immediately; never overwrite historical M0
    batch=record['batch'];emit('M1_BATCH_TERMINAL',dict(phase=name,**{k:batch.get(k) for k in ('id','status','request_counts','usage')}))
    if batch.get('status')!='completed':raise RuntimeError('batch terminal failure; no paid retry or replacement')
    items=batch.get('results') or [];byid={r.get('custom_id'):r for r in items}
    if len(items)!=len(byid) or set(byid)!={r['custom_id'] for r in rows}:raise RuntimeError('incomplete/duplicate IDs; retrieve only, never resubmit')
    cost=(batch.get('usage') or {}).get('cost')
    if cost is not None and (type(cost) not in (int,float) or not math.isfinite(cost) or cost<0):raise RuntimeError('invalid provider cost')
    return byid,dict(phase=name,batch_id=batch['id'],provider_cost_usd=cost,logical_requests=len(rows),
        paid_post_attempts=record['post_attempts'],get_attempts=record['get_attempts'],get_errors=record['get_errors'],
        started_unix=record['started_unix'],retrieved_unix=time.time(),first_run_id=record['first_run_id'],
        latency_scope='batch wall time including queue/poll/outage; per-request latency unavailable',
        payload_sha256=h,raw_batch_sha256=sha(json_text(batch).encode()))

def reconstruct(rows,raw,docs,cfg,ref):
    channels=('O_shared','O_extra','MP');events=[];ledgers={}
    for did,d in docs.items():
        for p in sorted(as_pairs(d['pairs'])):
            ledgers[(did,*p)]={c:dict(E_planned=0,E_submitted=0,E_delivered=0,E_valid=0,V=0,R=0) for c in channels}
    for r in rows:
        item=raw.get(r['custom_id']);event=decode_response(r,item or {},cfg)
        body=((item or {}).get('response') or {}).get('body') or {}
        event.update(doc_id=r['doc_id'],topic=r['topic'],channel=r['channel'],pass_index=r['channel_pass'],
            custom_id=r['custom_id'],prompt_sha256=r['prompt_sha256'],
            provider=body.get('provider'),system_fingerprint=body.get('system_fingerprint'),created=body.get('created'),
            response_id=body.get('id'),finish_reason=(body.get('choices') or [{}])[0].get('finish_reason'))
        events.append(event)
        for a,b in r['allowed_original']:
            l=ledgers[(r['doc_id'],*pair(a,b))][r['channel']]
            l['E_planned']+=1;l['E_submitted']+=int(item is not None)
            l['E_delivered']+=int(event['delivered']);l['E_valid']+=int(event['valid'])
        for a,b in event['positive']:ledgers[(r['doc_id'],*pair(a,b))][r['channel']]['V']+=1
        for a,b in event['explicit_reject']:ledgers[(r['doc_id'],*pair(a,b))][r['channel']]['R']+=1
    base={(did,*pair(*r['pair'])) for did,rr in ref.items() for r in rr if r['baseline_positive']}
    zero={(did,*pair(*r['pair'])) for did,rr in ref.items() for r in rr if r['luna_zero'] and not r['baseline_positive']}
    proposals={c:{k for k,l in ledgers.items() if l[c]['V']>0} for c in channels}
    admitted={c:s&zero for c,s in proposals.items()}
    encode=lambda s:[dict(doc_id=k[0],pair=list(k[1:])) for k in sorted(s)]
    coverage_out={did:coverage(d,[dict(pair=list(k[1:]),**l['MP']) for k,l in ledgers.items() if k[0]==did]) for did,d in docs.items()}
    assert all(l['MP']['E_planned']==1 for l in ledgers.values())
    assert all(l['O_shared']['E_planned']==7 and l['O_extra']['E_planned']==8 for l in ledgers.values())
    return dict(gold_used=False,raw_results=raw,events=events,
        ledgers=[dict(doc_id=k[0],pair=list(k[1:]),**l) for k,l in sorted(ledgers.items())],
        raw_proposed_channels={c:encode(s) for c,s in proposals.items()},admitted_channels={c:encode(s) for c,s in admitted.items()},
        final_predictions={'C':encode(base|admitted['O_shared']|admitted['O_extra']),'H':encode(base|admitted['O_shared']|admitted['MP'])},
        coverage_MP=coverage_out,integrity=dict(logical_requests=len(rows),delivered_responses=sum(e['delivered'] for e in events),
            states=dict(Counter(e['state'] for e in events)),all_planned_responses_valid=all(e['valid'] for e in events),
            shared_generated_once=True,shared_custom_ids=[r['custom_id'] for r in rows if r['channel']=='O_shared']))

def resources(events,batches,cfg):
    cc={}
    for channel in ('O_shared','O_extra','MP'):
        ee=[e for e in events if e['channel']==channel];pp=out=reason=cache=0;available=0;reason_available=cache_available=0
        for e in ee:
            u=e.get('usage') or {}
            if type(u.get('prompt_tokens')) is int and type(u.get('completion_tokens')) is int:
                available+=1;pp+=u['prompt_tokens'];out+=u['completion_tokens']
            reason+=(u.get('completion_tokens_details') or {}).get('reasoning_tokens',0) or 0
            cache+=(u.get('prompt_tokens_details') or {}).get('cached_tokens',0) or 0
            reason_available+=int(type((u.get('completion_tokens_details') or {}).get('reasoning_tokens')) is int)
            cache_available+=int(type((u.get('prompt_tokens_details') or {}).get('cached_tokens')) is int)
        cc[channel]=dict(logical_requests=len(ee),usage_available=available,reasoning_usage_available=reason_available,cache_usage_available=cache_available,prompt_tokens=pp,
            completion_tokens_including_reasoning=out,reasoning_tokens_included_in_completion=reason,cached_tokens=cache,
            weighted_token_index_usd=(pp*cfg['input_price_per_million']+out*cfg['output_price_per_million'])/1e6)
    complete=all(c['logical_requests']==c['usage_available'] for c in cc.values())
    def add(a,b):return {k:cc[a][k]+cc[b][k] for k in cc[a]}
    systems={'C':add('O_shared','O_extra'),'H':add('O_shared','MP')}
    ratio=systems['H']['weighted_token_index_usd']/systems['C']['weighted_token_index_usd'] if systems['C']['weighted_token_index_usd'] else None
    marginal=cc['MP']['weighted_token_index_usd']/cc['O_extra']['weighted_token_index_usd'] if cc['O_extra']['weighted_token_index_usd'] else None
    actual=[b['provider_cost_usd'] for b in batches]
    return dict(by_channel=cc,by_system_counterfactual=systems,batches=batches,usage_complete=complete,
        total_unique_logical_requests=sum(c['logical_requests'] for c in cc.values()),
        technical_paid_POSTs=sum(b['paid_post_attempts'] for b in batches),
        technical_GETs=sum(b['get_attempts'] for b in batches),
        actual_marginal_provider_cost_usd=sum(actual) if all(c is not None for c in actual) else None,
        provider_cost_allocation_by_system='unavailable unless per-request billing returned; weighted index only',
        weighted_H_over_C=ratio,weighted_MP_over_extra=marginal,
        cost_match_total_within5pct=complete and ratio is not None and abs(ratio-1)<=.05,
        cost_match_incremental_within5pct=complete and marginal is not None and abs(marginal-1)<=.05,
        reasoning_not_double_counted=True,
        cache_write_tokens=[(e.get('usage',{}).get('prompt_tokens_details') or {}).get('cache_write_tokens') for e in events])

def main():
    frozen=HERE/'frozen';protocol=json.loads((frozen/'m1-protocol-frozen.json').read_text());cfg=protocol['settings']
    runid=os.environ.get('ORX_RUN_ID')
    if not runid:raise RuntimeError('must run via tracked experiment launch, no direct inference')
    out=ROOT/'logs'/('m1-'+runid);out.mkdir(parents=True,exist_ok=True)
    for p in frozen.iterdir():
        if p.is_file():shutil.copyfile(p,out/p.name)
    emit('M1_CONFIG',dict(protocol=protocol,commit=os.environ.get('ORX_COMMIT'),run_id=runid))
    audit(out)
    rows=[json.loads(l) for l in (out/'m1-requests.jsonl').read_text().splitlines()]
    gate=catalog_gate(cfg,max(r['prompt_utf8_upper_bound'] for r in rows)+cfg['max_output_tokens'])
    budget=sum((r['prompt_utf8_upper_bound']*cfg['input_price_per_million']+cfg['max_output_tokens']*cfg['output_price_per_million'])/1e6 for r in rows)
    if budget>3 or len(rows)>528:raise RuntimeError('hard budget gate failed; no POST')
    state=Path('~/.local/share/openresearch/workere-decisions/m1-diversity-v1').expanduser()/sha((out/'m1-protocol-frozen.json').read_bytes())
    state.mkdir(parents=True,exist_ok=True)
    # Hold a protocol controller lock across phases so a concurrent resume cannot submit.
    with (state/'controller.lock').open('a') as controller:
        fcntl.flock(controller,fcntl.LOCK_EX|fcntl.LOCK_NB)
        reservation=state/'reservation.json'
        if not reservation.exists():save(reservation,dict(total_api_ceiling_usd=budget,cap_usd=3,logical_requests=len(rows),
            protocol_sha256=sha(out/'m1-protocol-frozen.json'),first_run_id=runid))
        prior=json.loads(reservation.read_text());assert prior['total_api_ceiling_usd']==budget
        a=json.loads((out/'m1-preflight.json').read_text());a.update(status='PASS',provider_budget_verified=True,provider_gate=gate,reservation=prior);save(out/'m1-preflight.json',a)
        emit('M1_PREFLIGHT',dict(status='PASS',reserved_api_ceiling_usd=budget,protocol_sha256=sha(out/'m1-protocol-frozen.json')))
        raw={};batches=[]
        for phase in ('shared','additional'):
            rr=[r for r in rows if (r['channel']=='O_shared')==(phase=='shared')]
            results,cost=paid_batch(rr,phase,state,protocol);raw.update(results);batches.append(cost)
            save(out/('m1-raw-'+phase+'.json'),results);save(out/'m1-batch-receipts.json',batches)
            export(out,['m1-raw-'+phase+'.json','m1-batch-receipts.json','m1-preflight.json'])
        docs=json.loads((out/'m1-public-documents.json').read_text());ref=json.loads((out/'m1-reference-without-gold.json').read_text())
        outputs=reconstruct(rows,raw,docs,cfg,ref)
        save(out/'m1-outputs-before-gold.json',outputs)
        receipt=dict(outputs_sha256=sha(out/'m1-outputs-before-gold.json'),protocol_sha256=sha(out/'m1-protocol-frozen.json'),
            frozen_unix=time.time(),commit=os.environ.get('ORX_COMMIT'),run_id=runid)
        save(out/'m1-output-freeze-receipt.json',receipt);emit('M1_FROZEN_BEFORE_GOLD',receipt)
        costs=resources(outputs['events'],batches,cfg);costs['reserved_api_ceiling_usd']=budget
        if costs['actual_marginal_provider_cost_usd'] is not None:assert costs['actual_marginal_provider_cost_usd']<=budget
        save(out/'m1-costs.json',costs)
        subprocess.run([sys.executable,str(HERE/'m1-analysis.py'),'--dir',str(out),'--repo',str(ROOT)],check=True)
        for name in ('m1-analysis.py','prepare.py','preflight.py','run.py'):shutil.copyfile(HERE/name,out/name)
        manifest=dict(commit=os.environ.get('ORX_COMMIT'),run_id=runid,python=sys.version,
            files_sha256={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='m1-verification-manifest.json'})
        save(out/'m1-verification-manifest.json',manifest);export(out)
        emit('M1_RESULT',json.loads((out/'m1-results.json').read_text()))

if __name__=='__main__':main()
