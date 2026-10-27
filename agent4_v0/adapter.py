"""Frozen Luna task adapter. Offline/dry-run by default; never substitutes a model.

Live transport is the original pinned BatchChatClient, imported ONLY after all gates.
Each task gets an isolated transport cache/receipt directory: repeated originals must
be fresh calls, not deduplicated batch/cache entries. No automatic paid retry in V0.
"""
import argparse
from collections import Counter
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import random
import sqlite3
import sys
import time

from eci.parse import parse_triples
from .common import digest, load, require, rows, jsonl, save, sha

SERVED = {'openai/gpt-6-luna', 'openai/gpt-6-luna:batch', 'openai/gpt-6-luna-20260922'}
REFERENCE_FILES = ('pipelines/jev/client.py','pipelines/jev/llm.py','pipelines/jev/batch.py',
                   'pipelines/jev/spend.py','eci/parse.py','eci/labels.py')


def markup(text, mentions):
    ordered = sorted(mentions, key=lambda m: (m['start'], m['end']))
    chunks, cursor = [], 0
    for m in ordered:
        require(cursor <= m['start'] < m['end'] <= len(text), 'overlapping or bad mention offsets')
        require(text[m['start']:m['end']] == m['surface'], 'mention surface mismatch')
        require(m['id'] and not any(c.isspace() or c in '<>,' for c in m['id']), 'invalid mention ID')
        chunks += [text[cursor:m['start']], '<'+m['id']+' '+m['surface']+'>']
        cursor = m['end']
    return ''.join(chunks) + text[cursor:]


def payload(task, snapshot, baseline):
    require(task['doc_id'] == snapshot['doc_id'] == baseline['doc_id'], 'document mismatch')
    require(digest(snapshot['request']) == snapshot['request_sha256'], 'snapshot tampered')
    import hashlib
    require(hashlib.sha256(task['document_text'].encode()).hexdigest() == task['text_sha256'], 'task text hash mismatch')
    original_ids = {m['id'] for m in baseline['mentions']}
    require(snapshot['demonstration_id'].split('_')[0] != task['doc_id'].split('_')[0], 'same-topic demo')
    mapping = task['id_map']
    require(set(mapping) == original_ids and len(set(mapping.values())) == len(original_ids), 'nonbijective IDs')
    require(len(task['mentions']) == len(baseline['mentions']), 'lost event mention')
    transformed = {m['id']: m['surface'] for m in task['mentions']}
    require(transformed == {mapping[m['id']]: m['surface'] for m in baseline['mentions']}, 'event surfaces changed')
    inverse = {v:k for k,v in mapping.items()}
    require(len(task['exposed_pair']) == 2 and set(task['exposed_pair']) <= set(inverse), 'invalid exposed pair')
    require(task['key'] == pilot_key(task['doc_id'],[inverse[m] for m in task['exposed_pair']]), 'task key/pair mismatch')
    expected = {tuple(sorted((mapping[p['a']], mapping[p['b']]))) for p in baseline['pairs']}
    require(len(task['candidate_pairs']) == len(expected) and
            {tuple(sorted(p)) for p in task['candidate_pairs']} == expected, 'candidate universe changed')
    old = markup(baseline['text'], baseline['mentions'])
    require(old == baseline['original_marked_text'], 'original markup not byte-identical to frozen suite')
    request = deepcopy(snapshot['request'])
    require(set(request) == {'model','messages','reasoning','max_tokens'}, 'unexpected frozen request fields')
    require(request['model'] == 'openai/gpt-6-luna:batch' and request['max_tokens'] == 32768 and
            request['reasoning'] == {'effort': 'high'}, 'frozen parameters changed')
    require([m['role'] for m in request['messages']] == ['system','user','assistant','user'], 'one demo required')
    last = request['messages'][-1]
    require(last['content'].count(old) == 1, 'frozen document splice is ambiguous')
    last['content'] = last['content'].replace(old, markup(task['document_text'], task['mentions']), 1)
    if task['variant'] in ('original','rename_ids'):
        require(task['document_text'] == baseline['text'], 'invariance changed lexical document')
    if task['variant'] == 'original':
        require(request == snapshot['request'], 'original request not identical to frozen request')
    return request


def prediction(task, response):
    record = {'task_id': task['task_id'], 'status': 'invalid_response'}
    try:
        require(isinstance(response, dict) and response.get('model') in SERVED, 'unexpected served model')
        require(len(response.get('choices', [])) == 1, 'one complete response required')
        choice = response['choices'][0]
        require(choice.get('finish_reason') == 'stop', 'truncated/non-complete response')
        text = choice['message']['content']
        require(isinstance(text, str), 'no string response')
        triples = parse_triples(text)  # EXACT reference parser; valid omission counts as zero.
        allowed = {tuple(p) for p in task['candidate_pairs']}
        allowed |= {(b,a) for a,b in allowed}
        last = {(a,b): lab for a,lab,b in triples if (a,b) in allowed and lab in ('causes','causedby','norel')}
        positive = sorted([list(p) for p,lab in last.items() if lab != 'norel'])
        a,b = task['exposed_pair']
        positive_set = {tuple(p) for p in positive}
        record.update(status='ok', predicted_ordered_pairs=positive,
                      vote_ab=int((a,b) in positive_set), vote_ba=int((b,a) in positive_set),
                      served_model=response['model'], provider=response.get('provider'),
                      response_id=response.get('id'), usage=response.get('usage', {}))
    except (ValueError, KeyError, TypeError) as exc:
        record['error'] = str(exc)
    return record


def reservation(body, budget):
    for key in ('input_price_per_million_upper','output_price_per_million_upper','fee_multiplier_upper'):
        require(type(budget.get(key)) in (int,float) and math.isfinite(budget[key]) and budget[key] > 0,
                'BLOCKED_BUDGET: missing finite positive verified price/fee upper bounds')
    # Conservative UTF-8 bytes + framing bound (not a measured tokenizer count).
    bound = len(json.dumps(body, ensure_ascii=False).encode()) + 1024
    require(bound + body['max_tokens'] <= budget['max_context_tokens'], 'context bound exceeded; never truncate')
    cost = (bound * budget['input_price_per_million_upper'] + body['max_tokens'] *
            max(.25, budget['output_price_per_million_upper'])) / 1e6 * max(1,budget['fee_multiplier_upper'])
    return cost, bound


class Ledger:
    """Atomic, conservative, NON-refundable attempt reservations; never reuse old project funds."""
    def __init__(self, path, limit, scope):
        require(type(limit) in (float,int) and math.isfinite(limit) and limit > 0, 'BLOCKED_BUDGET: invalid cap')
        self.db = sqlite3.connect(path)
        self.db.execute('CREATE TABLE IF NOT EXISTS scopes(scope TEXT PRIMARY KEY, cap REAL NOT NULL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS attempts(scope TEXT, task TEXT, usd REAL, PRIMARY KEY(scope,task))')
        self.db.execute('INSERT OR IGNORE INTO scopes VALUES (?,?)', (scope,limit))
        require(self.db.execute('SELECT cap FROM scopes WHERE scope=?',(scope,)).fetchone()[0] == limit,
                'BLOCKED_BUDGET: cap changed in existing scope')
        self.db.commit()
        self.scope, self.limit = scope, limit

    def reserve(self, task, cost):
        require(math.isfinite(cost) and cost > 0, 'invalid reservation')
        try:
            self.db.execute('BEGIN IMMEDIATE')
            require(not self.db.execute('SELECT 1 FROM attempts WHERE scope=? AND task=?',(self.scope,task)).fetchone(),
                    'attempt already reserved; inspect receipt, no automatic resubmission')
            count = self.db.execute('SELECT COUNT(*) FROM attempts WHERE scope=?',(self.scope,)).fetchone()[0]
            require(count < 84, 'BLOCKED_BUDGET: global V0 call limit exceeded')
            total = self.db.execute('SELECT COALESCE(SUM(usd),0) FROM attempts WHERE scope=?',(self.scope,)).fetchone()[0]
            require(total + cost <= self.limit, 'BLOCKED_BUDGET: atomic cap exceeded')
            self.db.execute('INSERT INTO attempts VALUES (?,?,?)',(self.scope,task,cost))
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise


def verify_review(tasks_path, baseline_path, selection_path, certificate):
    require(certificate is not None, 'BLOCKED_HUMAN_REVIEW: external review certificate missing')
    cert = load(certificate)
    require(cert.get('independent_human_attestation') is True, 'BLOCKED_HUMAN_REVIEW')
    for key,p in [('tasks',tasks_path),('baseline',baseline_path),('selection',selection_path)]:
        require(cert[key+'_sha256'] == sha(p), 'review certificate input mismatch: '+key)


def live(task, body, budget, reference_root, out, cost):
    # This import is deliberately unreachable from dry-run. No HTTP fallback is provided.
    root = Path(reference_root).resolve()
    require(set(budget.get('reference_source_sha256',{})) >= set(REFERENCE_FILES), 'frozen transport pins missing')
    for rel, expected in budget['reference_source_sha256'].items():
        require(sha(root/rel) == expected, 'frozen transport source changed: '+rel)
    require(sha(root/'eci/parse.py') == sha(Path(__file__).parents[1]/'eci/parse.py'), 'parser source differs')
    sys.path.insert(0,str(root))
    from pipelines.jev.batch import BatchChatClient
    path = out/'transport'/task['task_id']
    path.mkdir(parents=True,exist_ok=False)
    events=[]
    client = BatchChatClient(model=body['model'],model_version='gpt-6-luna-20260922',provider='openrouter',
        endpoint='https://openrouter.ai/api/v1/batches',api_key_env='OPENROUTER_API_KEY',mode='live',
        timeout=600,retries=0,requests_per_minute=60,max_calls=1,max_context_tokens=budget['max_context_tokens'],
        budget_usd=cost,max_cost_per_call_usd=cost,input_price_per_million=budget['input_price_per_million_upper'],
        cache_path=str(path/'cache.sqlite'),reasoning_effort='high',max_output_tokens=32768,
        shared_budget={'path':str(Path(budget['ledger_path']).with_suffix('.native.sqlite')),'scope':budget['scope'],
                       'limit_usd':budget['limit_usd'],'initial_spend_usd':0})
    try:
        require(client.validate_request(body['messages'],{})[0] == body, 'transport changes frozen request')
        def record(event):
            events.append(event)
            with (path/'events.jsonl').open('a') as f:
                f.write(json.dumps(event,ensure_ascii=False)+'\n')
        replies=client.decide_many([body['messages']],[record])
        return replies[0]['response'],events
    finally:
        client.close()


def run(a):
    out=Path(a.out)
    require(not out.exists(), 'adapter output already exists; use fresh directory')
    out.mkdir(parents=True)
    tasks=rows(a.tasks)
    baseline={x['doc_id']:x for x in rows(a.baseline)}
    selection=load(a.selection)
    require(sha(a.baseline) == selection['baseline_sha256'], 'baseline not sealed selection')
    selected={pilot_key(x['doc_id'],x['pair']) for x in selection['selected']}
    require(len(tasks) <= 84 and len(selection['selected']) <= 12, 'V0 budget universe exceeded')
    require(len({t['task_id'] for t in tasks}) == len(tasks), 'duplicate task IDs')
    require(all(t['variant'] in ('original','target','placebo_1','placebo_2','placebo_3','rename_ids') for t in tasks), 'unknown condition')
    counts=Counter((t['key'],t['variant']) for t in tasks)
    for key in {t['key'] for t in tasks}:
        require(key in selected, 'task outside selection')
        require({v:counts[key,v] for v in ('original','target','placebo_1','placebo_2','placebo_3','rename_ids')}
                == {'original':2,'target':1,'placebo_1':1,'placebo_2':1,'placebo_3':1,'rename_ids':1}, 'V0 read allocation differs')
    snaps={}
    for s in rows(a.snapshots):
        if s['pass_index'] == 0:
            require(s['doc_id'] not in snaps, 'duplicate frozen snapshot')
            snaps[s['doc_id']]=s
    blockers=[]
    snapshot_manifest = Path(getattr(a,'snapshots_manifest',None) or (Path(a.snapshots).parent/'snapshots-manifest.json'))
    if not snapshot_manifest.exists() or load(snapshot_manifest).get('snapshots_sha256') != sha(a.snapshots):
        blockers.append('BLOCKED_SNAPSHOTS: snapshot file not bound to sealed extraction manifest')
    try:
        verify_review(a.tasks,a.baseline,a.selection,a.review_certificate)
    except ValueError as exc:
        blockers.append(str(exc))
    if not tasks:
        blockers.append('NO_HUMAN_APPROVED_TASKS')
    budget=load(a.budget) if a.budget else None
    if not budget or budget.get('approved') is not True or not budget.get('authorized_by'):
        blockers.append('BLOCKED_BUDGET: no explicit signed-off envelope supplied')
    cap = budget.get('limit_usd') if budget else None
    cap_valid = type(cap) in (float,int) and math.isfinite(cap) and cap > 0
    if budget and not cap_valid:
        blockers.append('BLOCKED_BUDGET: missing finite positive monetary cap')
    if budget and (not isinstance(budget.get('ledger_path'),str) or not Path(budget['ledger_path']).is_absolute()):
        blockers.append('BLOCKED_BUDGET: persistent absolute ledger_path required')
    if budget and not budget.get('pricing_source_and_date'):
        blockers.append('BLOCKED_BUDGET: current pricing evidence missing')
    if budget and (not isinstance(budget.get('scope'),str) or not budget['scope'].startswith('agent4-v0-')):
        blockers.append('BLOCKED_BUDGET: new isolated agent4-v0 scope required')
    requests=[]
    random.Random('agent4-v0-condition-order-20261027').shuffle(tasks)
    total=0
    for t in tasks:
        if t['doc_id'] not in snaps:
            blockers.append('BLOCKED_SNAPSHOTS:'+t['doc_id'])
            continue
        s=snaps[t['doc_id']]
        body=payload(t,s,baseline[t['doc_id']])
        cost,bound=(None,None)
        if budget:
            try:
                cost,bound=reservation(body,budget)
                total+=cost
            except (ValueError,KeyError) as exc:
                blockers.append(str(exc))
        requests.append({'task_id':t['task_id'],'body':body,'request_sha256':digest(body),
                         'demo_id':s['demonstration_id'],'pass_index':0,'input_token_upper_bound':bound,
                         'cost_upper_usd':cost})
    jsonl(out/'requests.jsonl',requests)
    if cap_valid and total > cap:
        blockers.append('BLOCKED_BUDGET: total ceiling exceeds authorized cap')
    if a.execute and not (os.environ.get('OPENROUTER_API_KEY') or os.environ.get('OPENROUTER')):
        blockers.append('BLOCKED_MODEL_CREDENTIAL')
    save(out/'preflight.json',{'mode':'execute' if a.execute else 'dry_run',
        'blockers':sorted(set(blockers)), 'model_availability':'NOT_PROBED_NO_BILLABLE_HEALTH_CALL',
        'tasks':len(tasks),'request_bodies':len(requests),'requests_sha256':sha(out/'requests.jsonl'),
        'cost_upper_usd':total if budget and requests else None,'max_v0_calls':84,
        'stage':'before_execution','actual_model_calls_at_preflight':0,'actual_model_cost_usd_at_preflight':0,'paid_retries':0})
    if not a.execute:
        print(json.dumps({'mode':'dry_run','tasks':len(tasks),'blockers':sorted(set(blockers))}))
        return
    require(not blockers, 'PRECONDITIONS_BLOCKED: '+'; '.join(sorted(set(blockers))))
    require(a.reference_root, 'BLOCKED_FROZEN_TRANSPORT: --reference-root required')
    ledger_path=Path(budget['ledger_path'])
    ledger_path.parent.mkdir(parents=True,exist_ok=True)
    ledger=Ledger(ledger_path,budget['limit_usd'],budget['scope'])
    by_id={t['task_id']:t for t in tasks}
    # Append receipts BEFORE a possibly chargeable call. Unknown failures keep their reservation.
    with (out/'predictions.jsonl').open('x') as pred, (out/'call-log.jsonl').open('x') as log:
        for r in requests:
            ledger.reserve(r['task_id'],r['cost_upper_usd'])
            event={'task_id':r['task_id'],'request_sha256':r['request_sha256'],'demo_id':r['demo_id'],
                   'reservation_usd':r['cost_upper_usd'],'status':'reserved','started_at':time.time()}
            log.write(json.dumps(event)+'\n');log.flush()
            try:
                response,events=live(by_id[r['task_id']],r['body'],budget,a.reference_root,out,r['cost_upper_usd'])
                result=prediction(by_id[r['task_id']],response)
                result['demo_id']=r['demo_id']
                pred.write(json.dumps(result,ensure_ascii=False)+'\n');pred.flush()
                log.write(json.dumps({**event,'status':result['status'],'usage':response.get('usage'),
                                      'attempt_events':events},ensure_ascii=False)+'\n');log.flush()
                require(result['status']=='ok','invalid model output; stop, never default to zero')
            except Exception as exc:
                log.write(json.dumps({**event,'status':'failed_or_uncertain','error':str(exc)})+'\n');log.flush()
                raise


def pilot_key(doc,pair):
    return doc+'::'+'::'.join(sorted(pair))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('tasks','baseline','selection','snapshots','out'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--review-certificate')
    p.add_argument('--snapshots-manifest',help='default: snapshots-manifest.json beside snapshots')
    p.add_argument('--budget')
    p.add_argument('--reference-root')
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--dry-run',action='store_true',help='default; no transport imported')
    mode.add_argument('--execute',action='store_true')
    run(p.parse_args())


if __name__ == '__main__':
    main()
