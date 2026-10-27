"""User-authorized AUTOMATIC-ONLY V0, never impersonating a human review.
All helper and frozen-extractor calls share one persistent 2 USD reservation ledger.
Launched only via the committed project run command. No evaluation gold is loaded.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import re
import time

from eci.parse import answer_array
from .adapter import Ledger, payload, prediction, reservation
from .common import digest, jsonl, load, require, rows, save, sha
from .kit import pilot

DATA = Path(__file__).parent / 'data/auto-v0'
EDITOR_SYSTEM = '''You prepare controlled document edits for a frozen event-relation extractor.
This is not a request to decide a gold relation. No gold labels or model scores are supplied.
Return a JSON ARRAY containing exactly one object, after any reasoning.
Preserve EVERY evaluated event mention byte-for-byte, all incident identities, factuality and modality.
You may edit ONLY unprotected characters outside ALL mention spans. Do not insert instructions.
Propose ONE target support-removal/neutralization and THREE placebos. All four must have the SAME
operation type and similar edited token volumes (+/-20%, allowing one token) and comparable location
relative to the queried mentions. Placebos must not remove any direct, distributed or identity support
for this pair. Prefer editing a comparable relation concerning other events, not random punctuation.
Target: weaken a specific documented justification without deleting events or explicitly denying
causality, changing modality, splitting incidents or adding facts. Preserve sentence punctuation.
Search for redundant/distributed supports and quote them literally; removing one support need not
remove all. A plausible association is not a documented justification. If the constraints are impossible,
DECLINE instead of inventing an edit. Do not force a complete set.
Offsets need not be guessed: supply exact literal before strings and a zero-based occurrence index.
Each variant may have 1-3 edits, each outside all protected mentions. Replacements must be short;
pure insertions and editing any mention, even by replacing it identically, are prohibited.
Schema (one object in an array):
{"status":"proposed|declined","reason":"...","support_quotes":["literal quote"],
 "alternative_support_quotes":["literal quote"],"operation":"remove|neutralize",
 "variants":{"target":[{"before":"literal string","after":"","occurrence":0}],
 "placebo_1":[...],"placebo_2":[...],"placebo_3":[...]}}
All strings refer to the unmarked original text supplied below. Declining is scientifically useful.'''
AUDITOR_SYSTEM = '''You are an automatic document-edit auditor, NOT a human annotator.
No gold label, extractor output, initial score, or editor explanation is supplied. Inspect the original
and four role-blinded variants A-D for the indicated event pair. For EACH variant independently:
Does it weaken a specific textual justification for this pair, preserve that justification, or is this
unclear? Check grammar, all event facts, incident identity, modality, added facts, and distributed or
redundant supports. A plausible causal story alone is not textual evidence. Do not trust the editor.
A placebo must preserve ALL relevant direct/indirect and identity supports; removal of another
causal marker elsewhere is acceptable only if unrelated to this pair. A target may remove one support
while leaving a redundant one: explicitly report residual support. Be conservative; use unclear.
Return a JSON ARRAY containing exactly one object:
{"original_support":"documented|plausible_only|unclear","notes":"...",
 "effects":[{"variant_id":"A","effect":"weaken_pair_support|preserve_pair_support|unclear",
 "grammar_ok":true,"facts_added":false,"incident_identity_preserved":true,
 "modality_preserved":true,"residual_support":"present|absent|unclear","notes":"..."},...]}
Do not output a gold causal label or a confidence probability.'''


def one_object(response):
    arr=answer_array(response['choices'][0]['message']['content'])
    require(len(arr)==1 and isinstance(arr[0],dict),'helper output is not one JSON object in an array')
    return arr[0]


def locate(text, entry):
    before, after=entry['before'],entry['after']
    require(isinstance(before,str) and before and isinstance(after,str),'bad literal edit')
    require(not any(c in after for c in '<>\n\r'),'markup/newline injection')
    require(len(after)<=max(len(before)*2,80),'unbounded replacement')
    matches=[m.start() for m in re.finditer(re.escape(before),text)]
    occurrence=entry.get('occurrence',0)
    require(type(occurrence) is int and 0<=occurrence<len(matches),'literal before not found')
    a=matches[occurrence]
    return {'start':a,'end':a+len(before),'before':before,'after':after}


def structural(doc, proposal):
    require(proposal.get('status')=='proposed','editor_declined')
    require(proposal.get('operation') in ('remove','neutralize'),'unsupported intervention family')
    quotes=proposal.get('support_quotes',[])
    require(bool(quotes) and all(isinstance(q,str) and q and q in doc['text'] for q in quotes),'missing literal support')
    require(all(isinstance(q,str) and q in doc['text'] for q in proposal.get('alternative_support_quotes',[])),
            'invented alternative support quote')
    required=('target','placebo_1','placebo_2','placebo_3')
    require(set(proposal['variants'])==set(required),'not exactly four semantic variants')
    result={}
    for variant in required:
        raw=proposal['variants'][variant]
        require(isinstance(raw,list) and 1<=len(raw)<=3,'invalid edit count')
        edits=[locate(doc['text'],e) for e in raw]
        if proposal['operation']=='remove':
            require(all(not e['after'].strip() for e in edits),'operation type differs')
        else:
            require(all(e['after'].strip() for e in edits),'operation type differs')
        text,mentions=pilot.apply_edits(doc['text'],doc['mentions'],edits)
        require(text!=doc['text'],'no actual edit')
        require(re.findall(r'[.!?]',text)==re.findall(r'[.!?]',doc['text']),'sentence punctuation changed')
        result[variant]={'text':text,'mentions':mentions,'edits':edits,'weight':pilot.edit_weight(edits)}
    target=result['target']['weight']
    require(target>0 and all(abs(v['weight']-target)<=max(1,__import__('math').ceil(.2*target))
                             for v in result.values()),'volume matching fails')
    require(len({v['text'] for v in result.values()})==4,'duplicate target/placebo variants')
    return result


def audit_passes(audit, blind_map):
    require(audit.get('original_support')=='documented','original support not documented by automatic auditor')
    effects=audit.get('effects',[])
    require(len(effects)==4 and {e['variant_id'] for e in effects}==set(blind_map),'incomplete auditor verdict')
    by_variant={blind_map[e['variant_id']]:e for e in effects}
    for variant,e in by_variant.items():
        require(e.get('grammar_ok') is True and e.get('facts_added') is False and
                e.get('incident_identity_preserved') is True and e.get('modality_preserved') is True,
                'automatic semantic checks not satisfied:'+variant)
        expected='weaken_pair_support' if variant=='target' else 'preserve_pair_support'
        require(e.get('effect')==expected,'automatic role-blind effect mismatch:'+variant)
    return by_variant


def helper_body(system, public, max_tokens):
    return {'model':'openai/gpt-6-luna:batch','reasoning':{'effort':'high'},'max_tokens':max_tokens,
            'messages':[{'role':'system','content':system},
                        {'role':'user','content':json.dumps(public,ensure_ascii=False)}]}


def compile_tasks(item,doc,variants):
    mids=[m['id'] for m in doc['mentions']]
    rename=dict(zip(mids,mids[1:]+mids[:1]))
    result=[]
    key=pilot.keypair(item['doc_id'],item['pair'])
    for variant in pilot.CONDITIONS:
        text=doc['text'] if variant in ('original','rename_ids') else variants[variant]['text']
        mentions=doc['mentions'] if variant in ('original','rename_ids') else variants[variant]['mentions']
        mapping=rename if variant=='rename_ids' else {m:m for m in mids}
        for reading in range(2 if variant=='original' else 1):
            result.append({'task_id':digest([key,variant,reading,hashlib.sha256(text.encode()).hexdigest()])[:22],
             'key':key,'doc_id':item['doc_id'],'topic_id':item['topic_id'],'cohort':'positive',
             'variant':variant,'reading':reading,'document_text':text,
             'mentions':[{**m,'id':mapping[m['id']]} for m in mentions],
             'id_map':mapping,'exposed_pair':[mapping[m] for m in item['pair']],
             'candidate_pairs':[[mapping[p['a']],mapping[p['b']]] for p in doc['pairs']],
             'text_sha256':hashlib.sha256(text.encode()).hexdigest(),
             'validation_kind':'AUTOMATIC_PROVISIONAL_NOT_HUMAN_REVIEWED'})
    return result


def native_limits(costs):
    require(bool(costs),'empty native batch')
    slot=max(costs)+1e-12
    return slot, len(costs)*slot+1e-10


def require_never_submitted(stage_path, jobs=None):
    # The pinned SDK creates its durable receipt directory BEFORE any POST.
    require(not (stage_path/'luna-batch-jobs').exists() and not (stage_path/'events.jsonl').exists(),
            'stage may have been submitted; no automatic replay')
    plan=stage_path/'request-plan.json'
    require(plan.exists(),'no pre-POST request plan to reconcile')
    if jobs is not None:require(digest(load(plan)['jobs'])==digest(jobs),'pre-POST stage jobs changed')


class Calls:
    def __init__(self,out,budget):
        self.out,self.budget=out,budget
        self.ledger=Ledger(out/'budget.sqlite',2,budget['scope'])
        self.events=[]

    def batch(self,stage,jobs):
        """Dedup only within this wave; distinct waves NEVER share a generation cache."""
        dest=self.out/stage
        if (dest/'completed.json').exists():
            cached=load(dest/'completed.json')
            require(cached['jobs_sha256']==digest(jobs),'completed stage input changed')
            return cached['responses']
        recovered=dest.exists()
        if recovered:require_never_submitted(dest,jobs)
        else:dest.mkdir()
        unique={digest(j['body']):j['body'] for j in jobs}
        costs={h:reservation(body,self.budget)[0] for h,body in unique.items()}
        slot,native_total=native_limits(list(costs.values()))
        existing={}
        for h,cost in costs.items():
            old=self.ledger.db.execute('select usd from attempts where scope=? and task=?',
                                      (self.budget['scope'],stage+'::'+h)).fetchone()
            if old:
                require(recovered and old[0]==cost,'reservation changed or submitted stage cannot be replayed')
                existing[h]=old[0]
        spent=self.ledger.db.execute('select coalesce(sum(usd),0) from attempts where scope=?',(self.budget['scope'],)).fetchone()[0]
        require(spent+native_total-sum(existing.values())<=2,'2 USD cap cannot cover this whole stage')
        for h,cost in costs.items():
            if h not in existing:self.ledger.reserve(stage+'::'+h,cost)
        if not recovered:save(dest/'request-plan.json',{'jobs':jobs,'unique_requests':len(unique),'reservation_usd':sum(costs.values())})
        print('AGENT4_STAGE_SUBMIT '+json.dumps({'stage':stage,'requests':len(unique),
              'task_aliases':len(jobs),'reservation_usd':sum(costs.values())}),flush=True)
        from pipelines.jev.batch import BatchChatClient
        # All stage requests share max_tokens. No prompt or provider nonce is injected.
        require(len({b['max_tokens'] for b in unique.values()})==1,'mixed token caps')
        client=BatchChatClient(model='openai/gpt-6-luna:batch',model_version='gpt-6-luna-20260922',provider='openrouter',
            endpoint='https://openrouter.ai/api/v1/batches',api_key_env='OPENROUTER_API_KEY',mode='live',
            timeout=600,retries=0,requests_per_minute=60,max_calls=len(unique),max_context_tokens=1050000,
            budget_usd=native_total,max_cost_per_call_usd=slot,
            input_price_per_million=self.budget['input_price_per_million_upper'],cache_path=str(dest/'cache.sqlite'),
            reasoning_effort='high',max_output_tokens=next(iter(unique.values()))['max_tokens'],
            shared_budget={'path':str(self.out/'native-budget.sqlite'),'scope':self.budget['scope'],
                           'limit_usd':2,'initial_spend_usd':0})
        hashes=list(unique)
        def callback(h):
            def record(event):
                event={**event,'stage':stage,'body_sha256':h}
                with (dest/'events.jsonl').open('a') as f:f.write(json.dumps(event,ensure_ascii=False)+'\n')
            return record
        try:
            for body in unique.values():require(client.validate_request(body['messages'],{})[0]==body,'request changed by client')
            replies=client.decide_many([unique[h]['messages'] for h in hashes],[callback(h) for h in hashes])
        finally:
            client.close()
        responses={h:r['response'] for h,r in zip(hashes,replies)}
        aliased={j['id']:responses[digest(j['body'])] for j in jobs}
        save(dest/'completed.json',{'jobs_sha256':digest(jobs),'responses':aliased})
        return aliased


def main():
    config=load(DATA/'protocol.json')
    require(os.environ.get('OPENROUTER_API_KEY') or os.environ.get('OPENROUTER'),'model credential missing before any reservation')
    require(config['human_review_waived_by_user'] is True and config['budget_limit_usd']==2,'protocol not authorized')
    for name,h in config['inputs_sha256'].items():require(sha(DATA/name)==h,'sealed input changed:'+name)
    for name,h in config['source_files_sha256'].items():require(sha(name)==h,'sealed source changed:'+name)
    out=Path.home()/'.local/share/openresearch/workere-decisions/agent4-auto-v0-2usd'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():
        old=load(out/'protocol.json')
        if digest(old)!=digest(config):
            require(config.get('repair_of_protocol_sha256')==digest(old),'persistent protocol changed')
            for k in ('inputs_sha256','editor_system_sha256','auditor_system_sha256','budget_limit_usd','human_review_waived_by_user'):
                require(config[k]==old[k],'scientific settings changed during technical repair')
            require_never_submitted(out/'01-editor')
            require(not (out/'02-role-blind-auditor').exists(),'later stage exists; do not upgrade protocol')
            (out/'protocol.json').rename(out/'protocol-before-pre-post-budget-repair.json')
            save(out/'protocol.json',config)
    else:save(out/'protocol.json',config)
    if (out/'summary.json').exists():
        print('AGENT4_ALREADY_COMPLETE '+json.dumps(load(out/'summary.json')),flush=True);return
    budget=load(DATA/'budget-authorization.json')
    budget['ledger_path']=str(out/'budget.sqlite')
    budget['scope']='agent4-v0-user-2usd-20261027'
    budget['limit_usd']=2
    docs=pilot.parse_docs(DATA/'baseline.jsonl',{'dev'})
    selection=load(DATA/'selection.json')
    require(sha(DATA/'baseline.jsonl')==selection['baseline_sha256'],'baseline selection mismatch')
    snapshots={s['doc_id']:s for s in rows(DATA/'frozen-snapshots.jsonl') if s['pass_index']==0}
    eligible=[q for q in selection['selected'] if q['doc_id'] in snapshots]
    missing=[q for q in selection['selected'] if q['doc_id'] not in snapshots]
    print('AGENT4_PROTOCOL '+json.dumps({'hash':digest(config),'selected':len(selection['selected']),
          'snapshot_eligible':len(eligible),'missing_snapshots':len(missing),'budget':2,
          'validation':'automatic_provisional_only','no_gold':True,'out':str(out)}),flush=True)
    calls=Calls(out,budget)
    editor_jobs=[]
    for q in eligible:
        doc=docs[q['doc_id']]
        public={'doc_id':q['doc_id'],'queried_pair':q['pair'],'text':doc['text'],'protected_mentions':doc['mentions']}
        editor_jobs.append({'id':pilot.keypair(q['doc_id'],q['pair']),'body':helper_body(EDITOR_SYSTEM,public,16384)})
    editor_responses=calls.batch('01-editor',editor_jobs)
    variants,attrition,audit_jobs,blind_maps={},{},[],{}
    for q in eligible:
        key=pilot.keypair(q['doc_id'],q['pair']);doc=docs[q['doc_id']]
        try:
            proposed=one_object(editor_responses[key]);v=structural(doc,proposed)
            # Numeric position matching is an additional gate, not a semantic certificate.
            center=sum((m['start']+m['end'])/2 for m in doc['mentions'] if m['id'] in q['pair'])/2
            dist=lambda x:sum(abs((e['start']+e['end'])/2-center) for e in x['edits'])/len(x['edits'])
            dt=dist(v['target'])
            require(all(abs(dist(x)-dt)<=max(300,.25*len(doc['text'])) for x in v.values()),'position matching fails')
            variants[key]=v
            names=list(v);random.Random(key+'::blind-audit').shuffle(names)
            blind_maps[key]=dict(zip('ABCD',names))
            public={'doc_id':doc['doc_id'],'queried_pair':q['pair'],'original_text':doc['text'],
                'protected_mentions':doc['mentions'],
                'variants':[{'variant_id':letter,'text':v[name]['text']} for letter,name in blind_maps[key].items()]}
            audit_jobs.append({'id':key,'body':helper_body(AUDITOR_SYSTEM,public,8192)})
        except (ValueError,KeyError,TypeError) as exc:attrition[key]={'stage':'editor_or_structure','reason':str(exc)}
    save(out/'editor-structural-screen.json',{'variants':variants,'attrition':attrition,'blind_maps':blind_maps})
    tasks=[];audits={}
    if audit_jobs:
        audit_responses=calls.batch('02-role-blind-auditor',audit_jobs)
        for q in eligible:
            key=pilot.keypair(q['doc_id'],q['pair'])
            if key not in variants:continue
            try:
                audit=one_object(audit_responses[key]);audits[key]=audit
                audit_passes(audit,blind_maps[key])
                tasks.extend(compile_tasks(q,docs[q['doc_id']],variants[key]))
            except (ValueError,KeyError,TypeError) as exc:attrition[key]={'stage':'automatic_auditor','reason':str(exc)}
    save(out/'automatic-audit.json',{'verdicts':audits,'attrition':attrition,
         'validation_kind':'AUTOMATIC_PROVISIONAL_NOT_HUMAN_REVIEWED'})
    jsonl(out/'tasks.jsonl',tasks)
    raw={}
    if tasks:
        waves=[[t for t in tasks if not(t['variant']=='original' and t['reading']==1)],
               [t for t in tasks if t['variant']=='original' and t['reading']==1]]
        for index,wave in enumerate(waves):
            random.Random('agent4-frozen-wave-'+str(index)).shuffle(wave)
            jobs=[{'id':t['task_id'],'body':payload(t,snapshots[t['doc_id']],docs[t['doc_id']])} for t in wave]
            raw.update(calls.batch('03-extractor-wave'+str(index),jobs))
        preds=[prediction(t,raw[t['task_id']]) for t in tasks]
        jsonl(out/'predictions.jsonl',preds)
        from types import SimpleNamespace
        pilot.score(SimpleNamespace(tasks=str(out/'tasks.jsonl'),predictions=str(out/'predictions.jsonl'),out=str(out/'scores.jsonl')))
    events=[]
    for p in out.glob('*/events.jsonl'):events.extend(rows(p))
    known=sum(e['cost_usd'] for e in events if e.get('cost_usd') is not None)
    unknown=sum(e.get('cost_usd') is None for e in events)
    reserves=calls.ledger.db.execute('select coalesce(sum(usd),0) from attempts where scope=?',(budget['scope'],)).fetchone()[0]
    summary={'selected_pairs':len(selection['selected']),'snapshot_eligible_pairs':len(eligible),
      'missing_snapshot_pairs':len(missing),'automatic_provisional_pairs':len(tasks)//7,
      'attrition':attrition,'model_task_outputs':len(tasks),'actual_model_request_events':len(events),
      'actual_api_generation_requests':sum(bool(e.get('api_call')) and not e.get('cache_hit',False) for e in events),
      'known_cost_usd':known,'unknown_cost_events':unknown,'reserved_upper_usd':reserves,'budget_cap_usd':2,
      'decision':'INCONCLUSIVE', 'scope':'feasibility_only_automatic_validation_not_independent_confirmation',
      'human_review':False,'gold_used':False,'f1_computed':False,'protocol_sha256':digest(config)}
    save(out/'freeze-before-gold.json',{'files_sha256':{str(p.relative_to(out)):sha(p) for p in out.rglob('*')
         if p.is_file() and p.suffix in ('.json','.jsonl')},'gold_used':False})
    save(out/'summary.json',summary)
    print('AGENT4_FINAL '+json.dumps(summary),flush=True)


if __name__=='__main__':main()
