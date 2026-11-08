"""Tracked AP stages on the persistent Qwen API. No local GPU model or paid API."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shutil
import time

from steering import ap_protocol as p
from steering.ap_protocol import a, c, f, api

HOURS = {'AP0':1, 'AP1':6, 'AP2':5, 'AP3':2, 'AP4':8}


class BudgetStop(RuntimeError):
    pass


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(api.canonical(value)+'\n')
    temporary.replace(path)


def freeze(path, value):
    if path.exists():
        assert read(path) == value, f'frozen input differs: {path.name}'
    else:
        save(path, value)


def wait_for(path, predicate=lambda x: True, hours=8):
    deadline = time.time()+hours*3600
    while time.time() < deadline:
        try:
            value = read(path)
            if predicate(value): return value
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        print('WAIT', str(path), flush=True)
        time.sleep(30)
    raise RuntimeError(f'dependency unavailable after {hours}h: {path}')


@contextmanager
def lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                print('WAIT shared local API lease', flush=True)
                time.sleep(30)
        try: yield
        finally: fcntl.flock(handle, fcntl.LOCK_UN)


def stage_deadline(root, stage):
    with lock(root/'budget.lock'):
        path = root/'budget.json'
        state = read(path) if path.exists() else dict(start=time.time(), stages={})
        state['stages'].setdefault(stage, time.time())
        save(path, state)
        return min(state['start']+24*3600, state['stages'][stage]+HOURS[stage]*3600)


def check_budget(deadline):
    if time.time() >= deadline-60:
        raise BudgetStop('campaign or stage wall-time ceiling reached; checkpoints retained')


class Engine:
    def __init__(self, root, deadline, namespace='main'):
        from transformers import AutoTokenizer
        self.root, self.deadline = root, deadline
        contract = read(root/'AP0/api_contract.json')
        server = read(Path(os.environ['SERVER_METADATA']))
        assert server == contract['server'], 'server contract changed since C3'
        identity = c.model_identity()
        for field in ('config_sha256','tokenizer_sha256','tokenizer_config_sha256','vllm_version'):
            assert identity[field] == server['model_identity'][field]
        assert api.digest(identity['weights']) == server['weights_signature']
        self.model = contract['served_model']
        assert self.model == 'qwen3.8-27b'
        assert os.environ['API_BASE_URL'] == contract['base_url']
        self.client = api.Client(contract['base_url'], Path(os.environ['API_KEY_FILE']).read_text().strip(),
                                 root/'http-cache', api.digest(dict(contract=contract,replicate_namespace=namespace)))
        models, _, _ = self.client.call('/models')
        assert self.model in {x['id'] for x in models['data']}
        self.tokenizer = AutoTokenizer.from_pretrained(c.MODEL, trust_remote_code=True)
        self.size, self.workers = 16, 2
        test = self.tokenizer.apply_chat_template([{'role':'user','content':'Return JSON label N.'}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True, preserve_thinking=True)
        obj,_,_ = self.client.call('/../tokenize',dict(model=self.model,prompt=test,add_special_tokens=False))
        assert obj['tokens'] == self.tokenizer.encode(test,add_special_tokens=False)

    def call(self, body):
        check_budget(self.deadline)
        # Outstanding network work cannot consume an unbounded budget.
        self.client.timeout = max(1, min(900, self.deadline-time.time()-5))
        return self.client.call('/completions',body,cache=True)

    def batch(self, jobs):
        spec, thinking, answer = (jobs[0][key] for key in ('schema','thinking','answer_tokens'))
        assert all((j['schema'],j['thinking'],j['answer_tokens']) == (spec,thinking,answer) for j in jobs)
        tok = self.tokenizer
        prompts = [tok.apply_chat_template(j['messages'],tokenize=False,add_generation_prompt=True,
                    enable_thinking=True,preserve_thinking=True) for j in jobs]
        tokens = [tok.encode(text,add_special_tokens=False) for text in prompts]
        if any(len(t)+thinking+answer+16 > 32768 or len(t)+thinking+16 > 16384 for t in tokens):
            raise ValueError('AP prompt exceeds frozen server limits; no silent truncation')
        started = time.monotonic()
        traces, hit1, _ = self.call(api.payload(self.model,tokens,thinking))
        traces = api.ordered_choices(traces,len(jobs))
        final = [tok.encode(text+trace['text']+'\n</think>\n\n',add_special_tokens=False)
                 for text,trace in zip(prompts,traces)]
        if any(len(t)>16384 for t in final): raise ValueError('final prefix exceeds input limit')
        payload = api.payload(self.model,final,answer,'direct')
        payload['structured_outputs']['json'] = spec
        result, hit2, _ = self.call(payload)
        outputs = api.ordered_choices(result,len(jobs))
        rows = []
        for job, before, after, trace, output in zip(jobs,tokens,final,traces,outputs):
            obj = p.parse(output['text'],spec) if output['finish_reason']=='stop' else None
            rows.append(dict(key=job['key'],status='valid' if obj is not None else 'invalid_answer',parsed=obj,
                answer=output['text'],thought=trace['text'],thinking_forced=trace.get('stop_reason')!='</think>',
                thinking_finish=trace['finish_reason'],answer_finish=output['finish_reason'],
                prompt_tokens=len(before),total_input_tokens=len(before)+len(after),
                think_tokens=len(trace.get('token_ids') or tok.encode(trace['text'],add_special_tokens=False)),
                answer_tokens=len(output.get('token_ids') or tok.encode(output['text'],add_special_tokens=False)),
                run_id=os.environ.get('ORX_RUN_ID'),commit=os.environ.get('ORX_COMMIT')))
        return rows, dict(seconds=time.monotonic()-started,cache_hits=int(hit1)+int(hit2),keys=[j['key'] for j in jobs])

    def execute(self, jobs, out):
        out.mkdir(parents=True,exist_ok=True)
        a.write_jsonl(out/'requests.jsonl',jobs)
        expected = {j['key'] for j in jobs}
        assert len(expected)==len(jobs)
        path = out/'responses.jsonl'
        c.load_completed(path,expected)  # Recover only an interrupted last JSONL record.
        done = {r['key']:r for r in c.jsonl(path)} if path.exists() else {}
        assert len(done)==(len(c.jsonl(path)) if path.exists() else 0) and set(done)<=expected
        pending = [j for j in jobs if j['key'] not in done]
        # Group by schema/budget; two batches maximum are submitted at a time.
        groups = defaultdict(list)
        for j in pending: groups[api.digest([j['schema'],j['thinking'],j['answer_tokens']])].append(j)
        chunks = [rows[i:i+self.size] for _,rows in sorted(groups.items()) for i in range(0,len(rows),self.size)]
        with path.open('a',buffering=1) as sink, (out/'batches.jsonl').open('a',buffering=1) as ledger, \
             ThreadPoolExecutor(max_workers=self.workers) as pool:
            for start in range(0,len(chunks),self.workers):
                check_budget(self.deadline)
                futures = [pool.submit(self.batch,chunk) for chunk in chunks[start:start+self.workers]]
                errors = []
                for future in futures:
                    try:
                        rows,cost = future.result()
                        for row in rows:
                            sink.write(api.canonical(row)+'\n'); done[row['key']]=row
                        sink.flush(); ledger.write(api.canonical(cost)+'\n')
                        print('AP_PROGRESS',api.canonical(dict(step=out.name,done=len(done),total=len(jobs),**cost)),flush=True)
                    except Exception as exc: errors.append(exc)
                if errors: raise errors[0]
        assert set(done)==expected
        return done

    def smoke(self):
        jobs=[]
        for task in (*p.TASKS,'synthesis'):
            jobs.append(job([{'role':'user','content':'Return a JSON object of the requested schema; use unknown or N and empty strings.'}],
                            p.schema(task),64,256,dict(smoke=task)))
        rows=self.execute(jobs,self.root/'smoke')
        assert all(r['status']=='valid' for r in rows.values()), 'AP schema smoke failed'
        print('AP_SMOKE passed tokenizer and all subtask schemas',flush=True)


def job(messages, schema, thinking, answer_tokens, meta):
    value = dict(messages=messages,schema=schema,thinking=thinking,answer_tokens=answer_tokens,meta=meta,protocol=p.VERSION)
    value['key']=api.digest(value)
    return value


def judgments(engine, routes, method, docs, rules, out, nonce=''):
    rows=p.judgment_rows(routes,method,docs,rules,nonce)
    freeze(out/'judgment_requests.json',rows)
    jobs=[]
    if method=='decomposed':
        for row in rows:
            for task in p.TASKS:
                jobs.append(job(p.prompt(row,task,rules[row['dataset']]),p.schema(task),*p.budgets(task),
                                dict(judgment=row['key'],task=task)))
        observations=engine.execute(jobs,out/'observations')
        by_row=defaultdict(dict)
        for j in jobs:
            row=observations[j['key']]
            by_row[j['meta']['judgment']][j['meta']['task']]=row
        final_jobs=[]; invalid=set()
        for row in rows:
            values=by_row[row['key']]
            if any(values[t]['status']!='valid' for t in p.TASKS):
                invalid.add(row['key']);continue
            presented=row['pair'][::-1] if row['reverse'] else row['pair']
            cleaned={t:p.clean_observation(t,values[t]['parsed'],row['context'],presented) for t in p.TASKS}
            final_jobs.append(job(p.prompt(row,'synthesis',rules[row['dataset']],cleaned),p.schema('synthesis'),
                                  *p.budgets('synthesis'),dict(judgment=row['key'],task='synthesis')))
    else:
        invalid=set()
        final_jobs=[job(p.prompt(row,'direct',rules[row['dataset']]),p.schema('direct'),
                         *p.budgets('direct',method=='c3_direct'),dict(judgment=row['key'],task='direct')) for row in rows]
    finals=engine.execute(final_jobs,out/'final')
    by_key={j['meta']['judgment']:finals[j['key']] for j in final_jobs}
    results=[]
    for row in rows:
        answer=by_key.get(row['key'])
        valid=bool(answer and answer['status']=='valid' and row['key'] not in invalid)
        results.append(dict(**row,status='valid' if valid else 'invalid_answer',
            label=a.canonical_label(answer['parsed']['label'],row['reverse']) if valid else None))
    a.write_jsonl(out/'judgments.jsonl',results)
    return results


def source_data(source):
    manifest=read(source/'manifest.json')
    ctb,reference,esc,rules=c.inputs()
    saved=read(source/'baselines.json')
    base={dataset:{doc:a.unordered(saved[dataset]['n1_p3' if dataset=='CTB' else 'qwen_p2_sym'][doc])
                   for doc in ids} for dataset,ids in manifest['selected'].items()}
    return manifest['selected'],dict(CTB=ctb,ESC=esc),rules,base


def dump_predictions(path, predictions):
    save(path,{ds:{doc:sorted(values) for doc,values in docs.items()} for ds,docs in predictions.items()})


def costs(out):
    result=Counter()
    for path in out.rglob('responses.jsonl'):
        for row in c.jsonl(path):
            result['logical_responses']+=1;result[row['status']]+=1
            result['forced_thinking']+=int(row['thinking_forced'])
            for key in ('prompt_tokens','total_input_tokens','think_tokens','answer_tokens'): result[key]+=row[key]
    result['observed_batch_wall_seconds_sum']=sum(r['seconds'] for path in out.rglob('batches.jsonl') for r in c.jsonl(path))
    return dict(result)


def complete(out, report):
    report.update(status='complete',run_id=os.environ.get('ORX_RUN_ID'),commit=os.environ.get('ORX_COMMIT'),cost=costs(out))
    save(out/'report.json',report)
    print('AP_RESULT',api.canonical(report),flush=True)
    save(out/'terminal.json',dict(status='complete',report_sha256=c.sha(out/'report.json')))
    if os.environ.get('ORX_NOTIFY_FILE'):
        with open(os.environ['ORX_NOTIFY_FILE'],'a') as stream: stream.write(f'{out.name} terminé : rapport et décisions de suite disponibles.\n')


def ap0(root, source):
    wait_for(source/'report.json',lambda x:'api_contract' in x and 'api_wall_cost' in x)
    deadline=stage_deadline(root,'AP0');out=root/'AP0';out.mkdir(parents=True,exist_ok=True)
    original=read(source/'report.json')
    for name in ('manifest.json','requests.jsonl','routes.jsonl','baselines.json','upstream_cost.json',
                 'routing_diagnostic.json','generations.jsonl','api_contract.json'):
        target=out/name
        if target.exists(): assert c.sha(target)==c.sha(source/name)
        else: shutil.copy2(source/name,target)
    check_budget(deadline)
    a.score(out)
    replay=read(out/'report.json')
    for condition,value in original['conditions'].items():
        assert replay['conditions'][condition]['per_document']==value['per_document'], 'C3 per-document replay mismatch'
    assert original['baselines']==replay['baselines']
    selected,docs,rules,base=source_data(source)
    results={method:p.metrics(p.apply(base,p.consensus(p.c3_rows(source,method))),base,docs) for method in a.ARMS}
    chosen={ds:max(a.ARMS,key=lambda method:(results[method][ds]['f1'],method=='direct')) for ds in selected}
    save(out/'exposure.json',dict(independent_confirmation=False,known_evaluation=selected,
        CTB183='previous baseline/training-feature/evaluation pool; no certified untouched documents',
        ESC40='previous method-development pool; historical reserve register must be re-audited before confirmation',
        policy='No AP0–AP4 dataset is claimed held out. No evaluation labels used in routing.'))
    freeze(root/'contract.json',dict(protocol=p.VERSION,source_run=source.name,source_manifest_sha=c.sha(source/'manifest.json'),
        source_generation_sha=c.sha(source/'generations.jsonl'),selected=selected,hours=HOURS,total_hours=24,
        no_training=True,no_weight_edit=True,local_only=True,seed=p.SEED,
        data_hashes={str(path.relative_to(c.DATA)):c.sha(path) for path in c.DATA.rglob('*.json')}))
    complete(out,dict(exact_replay=True,selected=selected,c3=results,diagnostic_method=chosen,
        repair_candidate={ds:chosen[ds] if results[chosen[ds]][ds]['delta'][0]>0 else None for ds in selected},
        upstream_cost=original['upstream_cost'],scope='reused development'))


def ap1(root, source, method, deadline):
    selected,docs,rules,base=source_data(source);out=root/f'AP1_{method}'
    routes=c.jsonl(source/'routes.jsonl')
    engine=Engine(root,deadline);engine.smoke()
    result=judgments(engine,routes,method,docs,rules,out)
    predictions=p.apply(base,p.consensus(result))
    dump_predictions(out/'predictions.json',predictions)
    report=p.metrics(predictions,base,docs)
    complete(out,dict(method=method,selected=selected,metrics=report,routes=len(routes),directed_ctb=p.directed_ctb(result,docs),
        max_output_per_pair_orientation=3072,upstream_cost=read(source/'upstream_cost.json')))


def ap1_choice(root, source):
    reports={}
    for method in ('direct','decomposed'):
        terminal=wait_for(root/f'AP1_{method}/terminal.json')
        if terminal['status']=='complete': reports[method]=read(root/f'AP1_{method}/report.json')
    selected,docs,rules,base=source_data(source)
    if len(reports)==2:
        pred={m:p.apply(base,p.consensus(c.jsonl(root/f'AP1_{m}/judgments.jsonl'))) for m in reports}
        save(root/'ap1-comparison.json',dict(primary='decomposed minus direct, matched maximum output budget',
             comparison=p.metrics(pred['decomposed'],pred['direct'],docs)))
        print('AP1_COMPARISON',api.canonical(read(root/'ap1-comparison.json')),flush=True)
    choices={};repair={}
    for ds in selected:
        eligible=[m for m,r in reports.items() if r['metrics'][ds]['exploratory_gate']]
        method=max(eligible,key=lambda m:(reports[m]['metrics'][ds]['f1'],m=='direct')) if eligible else None
        choices[ds]=method or 'c3_direct'
        repair[ds]=method
    return choices,repair,reports


def ap2(root, source, c2_source, deadline):
    choices,repair,reports=ap1_choice(root,source)
    selected,docs,rules,base=source_data(source);out=root/'AP2'
    _,_,views=a.bundle(c2_source)
    routes=p.coverage_routes(selected,docs,base,views)
    freeze(out/'routes.json',routes);freeze(out/'choices.json',choices)
    print('AP2_PLAN',api.canonical(dict(choices=choices,repair_methods=repair,
        routes_by_dataset=dict(Counter(r['dataset'] for r in routes)),
        selection='frozen unanimous negatives; lexical reservation then uniform; no gold',
        additions_only=True)),flush=True)
    engine=Engine(root,deadline);results=[]
    for ds in selected:
        subset=[r for r in routes if r['dataset']==ds]
        results.extend(judgments(engine,subset,choices[ds],docs,rules,out/ds))
    a.write_jsonl(out/'judgments.jsonl',results)
    pred=p.apply(base,p.consensus(results),additions_only=True)
    dump_predictions(out/'predictions.json',pred)
    complete(out,dict(selected=selected,choices=choices,repair_methods=repair,metrics=p.metrics(pred,base,docs),directed_ctb=p.directed_ctb(results,docs),
        routes=len(routes),selection='half lexical then uniform without replacement, document/distance round-robin, no gold'))


def ap3(root, source, deadline):
    choices,repair,reports=ap1_choice(root,source)
    ap2_state=wait_for(root/'AP2/terminal.json')
    selected,docs,rules,base=source_data(source);out=root/'AP3'
    repairs={}
    ap0_report=read(root/'AP0/report.json')
    policy={}
    for ds in selected:
        if repair[ds]:
            rows=c.jsonl(root/f'AP1_{repair[ds]}/judgments.jsonl');policy[ds]=repair[ds]
        elif ap0_report['repair_candidate'][ds]:
            method=ap0_report['repair_candidate'][ds]
            rows=p.c3_rows(source,method);policy[ds]='c3_'+method
        else: rows=[];policy[ds]=None
        repairs.update({key:value for key,value in p.consensus(rows).items() if key[0]==ds})
    recovered={}
    if ap2_state['status']=='complete':
        coverage_report=read(root/'AP2/report.json')
        decisions=p.consensus(c.jsonl(root/'AP2/judgments.jsonl'))
        recovered={key:value for key,value in decisions.items() if coverage_report['metrics'][key[0]]['exploratory_gate']}
    repaired=p.apply(base,repairs)
    systems=dict(reference=base,repair=repaired,coverage=p.apply(base,recovered,True),combined=p.apply(repaired,recovered,True))
    results={name:p.metrics(pred,base,docs) for name,pred in systems.items()}
    winner={ds:max(systems,key=lambda name:(results[name][ds]['f1'],name=='reference',name=='repair')) for ds in selected}
    # Freeze choices before stability generation or any enlarged-set evaluation.
    freeze(out/'policy.json',dict(repair=policy,coverage=choices,winner=winner,
        coverage_enabled={ds:ap2_state['status']=='complete' and coverage_report['metrics'][ds]['exploratory_gate'] for ds in selected}))
    print('AP3_POLICY',api.canonical(read(out/'policy.json')),flush=True)
    for name,pred in systems.items(): dump_predictions(out/f'{name}.json',pred)
    # Technical reproducibility check: fresh nonce, rotated batch composition, fixed source and labels hidden.
    stability={};engine=Engine(root,deadline,namespace='fresh-rebatch-v1')
    for ds in selected:
        method=choices[ds]
        source_rows=(c.jsonl(root/f'AP1_{method}/judgments.jsonl') if method!='c3_direct' else p.c3_rows(source,'direct'))
        original=p.consensus([r for r in source_rows if r['dataset']==ds])
        keys=sorted(original,key=lambda k:api.digest([p.SEED,'stability',k]))[:32]
        routes=[dict(dataset=k[0],doc_id=k[1],pair=list(k[2]),baseline=k[2] in base[k[0]][k[1]]) for k in keys]
        fresh=judgments(engine,routes,method,docs,rules,out/f'stability_{ds}',nonce='fresh-rebatch-v1')
        new=p.consensus(fresh)
        stability[ds]=dict(pairs=len(keys),changed=sum(new[k]!=original[k] for k in keys),
                           label='technical stability only; not an independent evaluation')
    complete(out,dict(selected=selected,systems=results,policy=read(out/'policy.json'),stability=stability))


def screen_jobs(selected, docs, rules):
    jobs=[]
    for ds,ids in selected.items():
        for doc_id in ids:
            pairs=c.pairs(docs[ds][doc_id])
            for start in range(0,len(pairs),12):
                block=pairs[start:start+12]
                context=c.windows(docs[ds][doc_id]['text'],{e for pair in block for e in pair})
                for order in (0,1):
                    indices=list(range(len(block)))
                    if order: indices.sort(key=lambda i:api.digest([p.SEED,doc_id,start,i]))
                    jobs.append(job(c.make_messages(rules[ds],context,block,indices),c.schema(len(block)),4096,512,
                        dict(dataset=ds,doc_id=doc_id,pairs=block,order=order)))
    return jobs


def ap4(root,source,c2_source,work,deadline):
    ap3_report=wait_for(root/'AP3/report.json',lambda r:r.get('status')=='complete')
    initial,docs,rules,initial_base=source_data(source);out=root/'AP4'
    policy=ap3_report['policy'];ctb,reference,esc,_=c.inputs()
    if all(name=='reference' for name in policy['winner'].values()):
        complete(out,dict(skipped=True,reason='No useful candidate; preserve reference, no expansion inference.'));return
    # Conservative forecast uses actual AP throughput, includes screening and both orientations.
    elapsed=max(1,sum(read(path)['seconds'] for method in ('direct','decomposed')
                      for path in (root/f'AP1_{method}').glob('wall-*.json')))
    ap1_cost=sum(read(root/f'AP1_{m}/report.json').get('cost',{}).get('think_tokens',0)+
                 read(root/f'AP1_{m}/report.json').get('cost',{}).get('answer_tokens',0)
                 for m in ('direct','decomposed') if (root/f'AP1_{m}/report.json').exists())
    rate=max(1,ap1_cost/elapsed)
    full=p.extension_selection(initial,ctb,reference,esc,True)
    new_full={ds:[d for d in ids if d not in initial[ds]] for ds,ids in full.items()}
    n_pairs=sum(len(c.pairs(docs[ds][d])) for ds,ids in new_full.items() for d in ids)
    # Worst-case all pairs routed + coverage; account for screen, orientation and a 1.5 safety factor.
    expected=1.5*(sum(2*((len(c.pairs(docs[ds][d]))+11)//12)*4608 for ds,ids in new_full.items() for d in ids)+
                  n_pairs*2*3072)/rate
    selection_file=out/'selection.json'
    if selection_file.exists():
        selected=read(selection_file)['selected']
    else:
        selected=full if expected < deadline-time.time()-900 else p.extension_selection(initial,ctb,reference,esc,False)
        freeze(selection_file,dict(selected=selected,full_forecast_seconds=expected,observed_tokens_per_second=rate,
                                  independent_confirmation=False))
    additions={ds:[d for d in ids if d not in initial[ds]] for ds,ids in selected.items()}
    _,predictions=f.historical_baselines(selected,ctb,reference,esc,work)
    base={ds:{d:a.unordered(predictions[ds]['n1_p3' if ds=='CTB' else 'qwen_p2_sym'][d]) for d in ids} for ds,ids in selected.items()}
    engine=Engine(root,deadline)
    jobs=screen_jobs(additions,docs,rules);outputs=engine.execute(jobs,out/'screen')
    _,_,views=a.bundle(c2_source)
    for j in jobs:
        r=outputs[j['key']];meta=j['meta']
        if r['status']=='valid':
            views[(meta['dataset'],meta['doc_id'],meta['order'])]={**views.get((meta['dataset'],meta['doc_id'],meta['order']),{}),
                **{tuple(pair):r['parsed'][str(i)]!='N' for i,pair in enumerate(meta['pairs'])}}
    result={ds:{d:set(base[ds][d]) for d in ids} for ds,ids in selected.items()}
    # Old subset uses the selected cached system exactly, not new draws.
    for ds in selected:
        old=read(root/f"AP3/{policy['winner'][ds]}.json")[ds]
        for d in initial[ds]: result[ds][d]=set(map(tuple,old[d]))
    for ds in selected:
        mode=policy['winner'][ds]
        if mode in ('repair','combined'):
            routes=[]
            for d in additions[ds]:
                candidates=a.route_pairs(set(c.pairs(docs[ds][d])),base[ds][d],[views.get((ds,d,o),{}) for o in (0,1)])
                routes.extend(dict(dataset=ds,doc_id=d,pair=list(pair),baseline=pair in base[ds][d]) for pair in sorted(candidates))
            method=policy['repair'][ds]
            if method=='c3_evidence':
                rows=evidence_judgments(engine,routes,docs,rules,out/f'repair_{ds}')
            else: rows=judgments(engine,routes,method,docs,rules,out/f'repair_{ds}')
            result=p.apply(result,p.consensus(rows))
        if mode in ('coverage','combined'):
            routes=p.coverage_routes({ds:additions[ds]},docs,base,views,
                                    {ds:(128 if ds=='CTB' else 256)*len(additions[ds])//len(initial[ds])})
            rows=judgments(engine,routes,policy['coverage'][ds],docs,rules,out/f'coverage_{ds}')
            result=p.apply(result,p.consensus(rows),True)
    dump_predictions(out/'predictions.json',result)
    expanded=p.metrics(result,base,docs)
    only_new=p.metrics({ds:{d:result[ds][d] for d in ids} for ds,ids in additions.items()},
                       {ds:{d:base[ds][d] for d in ids} for ds,ids in additions.items()},docs)
    complete(out,dict(selected=selected,metrics=expanded,new_documents_only=only_new,policy=policy,
        scope='expanded reused development, not independent confirmation',upstream_cost='historical vote plus cached C2 and new screen; see AP0 and screen ledger'))


def evidence_judgments(engine,routes,docs,rules,out):
    rows=p.judgment_rows(routes,'c3_evidence',docs,rules);jobs=[]
    for row in rows:
        presented=row['pair'][::-1] if row['reverse'] else row['pair']
        jobs.append(job(a.messages(rules[row['dataset']],row['context'],presented,'evidence'),
                        a.answer_schema('evidence'),1024,512,dict(judgment=row['key'])))
    outputs=engine.execute(jobs,out/'evidence');result=[]
    for row,j in zip(rows,jobs):
        value=outputs[j['key']];presented=row['pair'][::-1] if row['reverse'] else row['pair']
        valid=value['status']=='valid' and a.grounded(value['parsed'],row['context'],presented)
        result.append(dict(**row,status='valid' if valid else 'invalid_answer',
            label=a.canonical_label(value['parsed']['label'],row['reverse']) if valid else None))
    a.write_jsonl(out/'judgments.jsonl',result);return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=('AP0','AP1_direct','AP1_decomposed','AP2','AP3','AP4'),required=True)
    args=parser.parse_args();stage=args.stage
    work=Path(os.environ['WORK']);root=work/'out/AP'/os.environ['AP_GROUP']
    source=work/'out/C3_API'/os.environ['C3_API_RUN'];c2_source=work/'out/C2'/os.environ['C2_RUN']
    out=root/stage;out.mkdir(parents=True,exist_ok=True)
    print('AP_CONFIG',api.canonical(dict(stage=stage,root=str(root),source=str(source),protocol=p.VERSION,
        run=os.environ.get('ORX_RUN_ID'),commit=os.environ.get('ORX_COMMIT'),hours=HOURS,total_hours=24)),flush=True)
    if (out/'terminal.json').exists() and read(out/'terminal.json')['status']=='complete':
        print('ALREADY_COMPLETE',str(out),flush=True);return
    if stage=='AP0': ap0(root,source);return
    wait_for(root/'AP0/terminal.json',lambda r:r['status']=='complete')
    if stage in ('AP2','AP3','AP4'):
        for method in ('direct','decomposed'): wait_for(root/f'AP1_{method}/terminal.json')
    if stage in ('AP3','AP4'): wait_for(root/'AP2/terminal.json')
    if stage=='AP4':
        dependency=wait_for(root/'AP3/terminal.json')
        if dependency['status']!='complete':
            complete(out,dict(skipped=True,reason='AP3 did not finish; no frozen expansion policy.'));return
    contract=read(root/'contract.json')
    assert contract['source_generation_sha']==c.sha(source/'generations.jsonl')
    for name,expected in contract['data_hashes'].items(): assert c.sha(c.DATA/name)==expected
    with lock(work/'ap-api.lock'):
        deadline=stage_deadline(root,stage.split('_')[0])
        started=time.time()
        save(out/f"attempt-{os.environ.get('ORX_RUN_ID','test')}.json",dict(start=started,deadline=deadline,commit=os.environ.get('ORX_COMMIT')))
        try:
            check_budget(deadline)
            if stage.startswith('AP1_'): ap1(root,source,stage.split('_',1)[1],deadline)
            elif stage=='AP2': ap2(root,source,c2_source,deadline)
            elif stage=='AP3': ap3(root,source,deadline)
            else: ap4(root,source,c2_source,work,deadline)
        except BudgetStop as exc:
            save(out/'terminal.json',dict(status='budget_exhausted',reason=str(exc),cost=costs(out)))
            print('AP_BUDGET_STOP',str(exc),flush=True)
        finally:
            save(out/f"wall-{os.environ.get('ORX_RUN_ID','test')}.json",dict(start=started,end=time.time(),seconds=time.time()-started))


if __name__=='__main__': main()
