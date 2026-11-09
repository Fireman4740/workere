"""Tracked validation of cached-vote combinations on CTB47, ESC40 and MAVEN12."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import time

from steering import ap_campaign as ap
from steering import ap_protocol as p
from steering import campaign_adjudicate as a
from steering import campaign_api as api
from steering import campaign_blocks as c
from steering import campaign_facts as f
from steering import validation_protocol as v

HOURS = {'CTB': 8, 'ESC': 16, 'MAVEN': 12}
MAVEN_REPO = 'Nofing/MAVEN-ERE-standard-eci-workere'
MAVEN_REVISION = '830a58ee7a092d5ca27b4a8c28c18e2c82abf814'


def incremental_cost(out):
    costs = Counter()
    for path in out.rglob('responses.jsonl'):
        for row in c.jsonl(path):
            if row.get('reused_from'):
                costs['reused_responses'] += 1
                continue
            costs['logical_responses'] += 1
            costs[row['status']] += 1
            costs['forced_thinking'] += int(row['thinking_forced'])
            for key in ('prompt_tokens', 'total_input_tokens', 'think_tokens', 'answer_tokens'):
                costs[key] += row[key]
    return dict(costs)


def import_responses(jobs, out, catalog):
    out.mkdir(parents=True, exist_ok=True)
    path = out/'responses.jsonl'
    expected = {j['key'] for j in jobs}
    completed = c.load_completed(path, expected)
    imported = 0
    with path.open('a') as stream:
        for job in jobs:
            item = catalog.get(v.cache_identity(job))
            if job['key'] in completed or item is None:
                continue
            row, source = item
            stream.write(api.canonical(dict(row, key=job['key'], reused_from=source))+'\n')
            imported += 1
    print('VC_REUSE', api.canonical(dict(stage=out.name, existing=len(completed), imported=imported,
                                        expected=len(jobs))), flush=True)


def cache_catalog(old, c2):
    """Native C2 and API AP answers stay labelled by their original numerical protocol."""
    catalog = {}
    def add(job, row, path):
        key = v.cache_identity(job)
        # Deterministic priority: first source wins; never pick answers by correctness.
        catalog.setdefault(key, (row, str(path)))
    rows = {r['key']: r for r in c.jsonl(c2/'generations.jsonl')}
    for req in c.jsonl(c2/'requests.jsonl'):
        if req['condition'] == 'plain_4096':
            add(ap.job(req['messages'], c.schema(len(req['pairs'])), 4096, 384,
                       dict(dataset=req['dataset'], doc_id=req['doc_id'], order=req['schedule'])),
                rows[req['key']], c2/'generations.jsonl')
    sources = [old/'AP4/screen', old/'AP1_direct/final', old/'AP4/repair_ESC/final']
    for source in sources:
        if not (source/'requests.jsonl').exists():
            continue
        outputs = {r['key']: r for r in c.jsonl(source/'responses.jsonl')}
        for job in c.jsonl(source/'requests.jsonl'):
            add(job, outputs[job['key']], source/'responses.jsonl')
    return catalog


def existing_screen_jobs(selected, docs, rules, old, c2):
    """Retain old panels exactly; use original C2 layout only for new documents."""
    result = []
    existing = set()
    for req in c.jsonl(c2/'requests.jsonl'):
        ds, doc_id = req['dataset'], req['doc_id']
        if req['condition'] != 'plain_4096' or doc_id not in selected.get(ds, []):
            continue
        result.append(ap.job(req['messages'], c.schema(len(req['pairs'])), 4096, 384,
            dict(dataset=ds, doc_id=doc_id, pairs=req['pairs'], order=req['schedule'],
                 layout='C2_original', block=req.get('block'))))
        existing.add((ds, doc_id))
    for job in c.jsonl(old/'AP4/screen/requests.jsonl'):
        meta = job['meta']
        if meta['doc_id'] in selected.get(meta['dataset'], []):
            assert (meta['dataset'], meta['doc_id']) not in existing
            result.append(job)
    covered = {(j['meta']['dataset'], j['meta']['doc_id']) for j in result}
    missing = {ds: [doc for doc in ids if (ds, doc) not in covered] for ds, ids in selected.items()}
    result.extend(v.screen_jobs(missing, v.public_docs(docs), rules))
    return sorted(result, key=lambda j: (j['meta']['dataset'], j['meta']['doc_id'],
                                        j['meta']['order'], v.fingerprint(j)))


def prepare_cached(dataset, old, c2, work, root, out):
    ctb, reference, esc, rules = c.inputs()
    selected = {dataset: list(reference['qwen_ids']) if dataset == 'CTB' else sorted(esc)}
    docs = {dataset: ctb if dataset == 'CTB' else esc}
    # Loader expects both dataset keys, but only requested documents are processed.
    _, history = f.historical_baselines(dict(CTB=selected.get('CTB', []), ESC=selected.get('ESC', [])),
                                         ctb, reference, esc, work)
    base = {dataset: {d: a.unordered(history[dataset]['n1_p3' if dataset == 'CTB' else 'qwen_p2_sym'][d])
                      for d in selected[dataset]}}
    initial = ap.read(c2/'manifest.json')['selected'][dataset]
    screen = existing_screen_jobs(selected, docs, rules, old, c2)
    manifest = dict(protocol=v.VERSION, dataset=dataset, selected=selected, initial=initial,
        dataset_sha256=c.sha(c.DATA/('ctb-suite.json' if dataset == 'CTB' else 'esc/test-s40.json')),
        rules_sha256=api.digest(rules[dataset]), baseline_sha256=api.digest({d: sorted(ps) for d, ps in base[dataset].items()}),
        screen_sha256=api.digest(screen), policies=list(v.POLICIES),
        source_hashes={str(path): c.sha(path) for path in [c2/'requests.jsonl', c2/'generations.jsonl',
                     old/'AP4/report.json', old/'AP4/screen/requests.jsonl', old/'AP4/screen/responses.jsonl',
                     work/'out/N1/gen.jsonl', work/'out/P3/gen.jsonl', work/'out/P2/gen.jsonl']},
        scope='reused development, binary presence only; initial native and later API caches explicitly mixed',
        no_training=True, no_weight_edit=True, local_only=True)
    ap.freeze(out/'manifest.json', manifest)
    ap.freeze(root/'AP0/api_contract.json', ap.read(old/'AP0/api_contract.json'))
    ap.dump_predictions(out/'baseline.json', base)
    print('VC_PLAN', api.canonical(manifest), flush=True)
    return selected, docs, rules, base, screen, initial


def preflight(engine, jobs):
    lengths = []
    for job in jobs:
        text = engine.tokenizer.apply_chat_template(job['messages'], tokenize=False,
                    add_generation_prompt=True, enable_thinking=True, preserve_thinking=True)
        n = len(engine.tokenizer.encode(text, add_special_tokens=False))
        assert n+job['thinking']+32 <= 16384, 'input/re-prefill limit exceeded, no document dropped'
        assert n+job['thinking']+job['answer_tokens']+32 <= 32768
        lengths.append(n)
    print('VC_PREFLIGHT', api.canonical(dict(requests=len(jobs), max_prompt=max(lengths, default=0),
                                             prompt_tokens=sum(lengths))), flush=True)


def repair_jobs(routes, docs, rules):
    rows = []
    for route in routes:
        for reverse in (False, True):
            row = dict(dataset=route['dataset'], doc_id=route['doc_id'], pair=route['pair'],
                baseline=route['baseline'], method='direct', reverse=reverse,
                context=v.source_window(docs[route['dataset']][route['doc_id']]['text'], set(route['pair'])), nonce='')
            row['key'] = api.digest(row)
            rows.append(row)
    rows.sort(key=lambda row: api.digest([p.SEED, row['key']]))
    jobs = [ap.job(p.prompt(row, 'direct', rules[row['dataset']]), p.schema('direct'),
                   2560, 512, dict(judgment=row['key'], task='direct')) for row in rows]
    return rows, jobs


def repair_decisions(rows, jobs, outputs):
    judgments = []
    for row, job in zip(rows, jobs, strict=True):
        answer = outputs[job['key']]
        obj = a.parse_final(answer.get('answer'), 'direct') if answer['status'] == 'valid' else None
        judgments.append(dict(row, status='valid' if obj else 'invalid_answer',
            label=a.canonical_label(obj['label'], row['reverse']) if obj else None))
    return p.consensus(judgments), judgments


def subset(predictions, ds, ids):
    return {ds: {d: predictions[ds][d] for d in ids}}


def evaluate(systems, base, docs, selected, initial):
    ds = next(iter(selected))
    reports = {name: p.metrics(pred, base, docs)[ds] for name, pred in systems.items()}
    extra = [d for d in selected[ds] if d not in initial]
    splits = {}
    for label, ids in [('initial', initial), ('extension', extra)]:
        if ids:
            splits[label] = {name: p.metrics(subset(pred, ds, ids), subset(base, ds, ids), docs)[ds]
                              for name, pred in systems.items()}
    return reports, splits


def finish(out, report):
    report.update(status='complete', run_id=os.environ.get('ORX_RUN_ID'), commit=os.environ.get('ORX_COMMIT'),
                  cost=incremental_cost(out), cost_note='Incremental only; historical votes and imported C2/AP responses are sunk upstream cost, not free deployment.')
    ap.save(out/'report.json', report)
    ap.save(out/'terminal.json', dict(status='complete', report_sha256=c.sha(out/'report.json')))
    print('VC_RESULT', api.canonical(report), flush=True)


def screen_repeat(engine, root, jobs, outputs, out, deadline):
    # Exact prompts on a small fixed panel, fresh HTTP namespace, no answer selection.
    chosen = sorted(jobs, key=lambda j: api.digest([v.VERSION, 'repeat', v.fingerprint(j)]))[:8]
    fresh_engine = ap.Engine(root, deadline, namespace=v.VERSION+'-fresh-'+out.name)
    fresh_engine.size = 4
    fresh = fresh_engine.execute(chosen, out/'repeat')
    changed = total = 0
    for job in chosen:
        old = c.parse_answer(outputs[job['key']].get('answer'), len(job['meta']['pairs']))
        new = c.parse_answer(fresh[job['key']].get('answer'), len(job['meta']['pairs']))
        if old is not None and new is not None:
            changed += sum((old[k] != 'N') != (new[k] != 'N') for k in old)
            total += len(old)
    return dict(requests=len(chosen), valid_paired_decisions=total, changed_binary_decisions=changed,
                scope='technical native/API or API/rebatch audit; not independent accuracy replication')


def run_cached(dataset, root, old, c2, work, out, deadline):
    selected, docs, rules, base, jobs, initial = prepare_cached(dataset, old, c2, work, root, out)
    catalog = cache_catalog(old, c2)
    import_responses(jobs, out/'screen', catalog)
    # Before any inference, verify the old CTB filter exactly.
    if dataset == 'CTB':
        initial_jobs = [j for j in jobs if j['meta']['doc_id'] in initial]
        cached = {r['key']: r for r in c.jsonl(out/'screen/responses.jsonl')}
        assert all(j['key'] in cached for j in initial_jobs), 'initial C2 panel must be cached'
        views = v.views_from(initial_jobs, cached)
        replay = v.combine(subset(base, dataset, initial), views, {})
        result, _ = evaluate(replay, subset(base, dataset, initial), docs, {dataset: initial}, initial)
        assert (result['B_filter']['tp'], result['B_filter']['fp'], result['B_filter']['fn']) == (26, 8, 18)
        print('VC_EXACT_C2_REPLAY', api.canonical({n: r['f1'] for n, r in result.items()}), flush=True)
    engine = ap.Engine(root, deadline, namespace=v.VERSION+'-'+dataset)
    preflight(engine, jobs)
    smoke = [ap.job([{'role': 'user', 'content': 'Return JSON {"0":"N"}.'}], c.schema(1), 64, 128,
                     dict(validation_smoke=dataset))]
    assert all(r['status'] == 'valid' for r in engine.execute(smoke, out/'smoke').values())
    outputs = engine.execute(jobs, out/'screen')
    views = v.views_from(jobs, outputs)
    routes = v.routes_for(selected, v.public_docs(docs), base, views)
    ap.freeze(out/'routes.json', routes)
    rows, final_jobs = repair_jobs(routes, docs, rules)
    preflight(engine, final_jobs)
    import_responses(final_jobs, out/'repair/final', catalog)
    print('VC_ROUTES', api.canonical(dict(dataset=dataset, pairs=len(routes), orientations=len(final_jobs))), flush=True)
    finals = engine.execute(final_jobs, out/'repair/final')
    decisions, judgments = repair_decisions(rows, final_jobs, finals)
    a.write_jsonl(out/'repair/judgments.jsonl', judgments)
    systems = v.combine(base, views, decisions)
    metrics, splits = evaluate(systems, base, docs, selected, initial)
    if dataset == 'ESC':
        previous = ap.read(old/'AP4/report.json')
        old_ids = previous['selected']['ESC']
        replay = p.metrics(subset(systems['C_repair'], dataset, old_ids), subset(base, dataset, old_ids), docs)['ESC']
        assert replay['per_document'] == previous['metrics']['ESC']['per_document'], 'AP4 old20 direct-repair replay differs'
        print('VC_EXACT_AP4_REPLAY', api.canonical({k: replay[k] for k in ('documents','tp','fp','fn','f1')}), flush=True)
    for name, predictions in systems.items():
        ap.dump_predictions(out/f'{name}.json', predictions)
    repeat = screen_repeat(engine, root, jobs, outputs, out, deadline)
    finish(out, dict(dataset=dataset, selected=selected, policies=metrics, splits=splits, repeatability=repeat,
                     manifest_sha256=c.sha(out/'manifest.json'), scope='expanded reused development; no held-out confirmation'))


class DocumentEngine(ap.Engine):
    """Same two-stage transport as AP, with strict array validation for document outputs."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Long document thoughts exceeded the old 900s timeout with 32 sequences.
        # Reuse completed reads, then process at most 8 remaining sequences.
        self.size, self.workers = 4, 2
        print('VC_DOCUMENT_TRANSPORT', api.canonical(dict(batch_size=self.size, workers=self.workers,
            max_concurrent_sequences=8, timeout_ceiling_seconds=3600,
            note='completed reads retained; pending reads use smaller batches after the first timeout')), flush=True)

    def call(self, body):
        ap.check_budget(self.deadline)
        self.client.timeout = max(1, min(3600, self.deadline-time.time()-5))
        return self.client.call('/completions', body, cache=True)

    def batch(self, jobs):
        spec, thinking, answer = (jobs[0][k] for k in ('schema', 'thinking', 'answer_tokens'))
        assert all((j['schema'], j['thinking'], j['answer_tokens']) == (spec, thinking, answer) for j in jobs)
        tok = self.tokenizer
        prompts = [tok.apply_chat_template(j['messages'], tokenize=False, add_generation_prompt=True,
                    enable_thinking=True, preserve_thinking=True) for j in jobs]
        before = [tok.encode(text, add_special_tokens=False) for text in prompts]
        assert all(len(t)+thinking+32 <= 16384 and len(t)+thinking+answer+32 <= 32768 for t in before)
        started = time.monotonic()
        traces, hit1, _ = self.call(api.payload(self.model, before, thinking))
        traces = api.ordered_choices(traces, len(jobs))
        after = [tok.encode(text+trace['text']+'\n</think>\n\n', add_special_tokens=False)
                 for text, trace in zip(prompts, traces, strict=True)]
        assert all(len(t) <= 16384 for t in after)
        body = api.payload(self.model, after, answer, 'direct')
        body['structured_outputs']['json'] = spec
        result, hit2, _ = self.call(body)
        finals = api.ordered_choices(result, len(jobs))
        rows = []
        for job, b, af, trace, final in zip(jobs, before, after, traces, finals, strict=True):
            obj = v.document_parse(final['text']) if final['finish_reason'] == 'stop' else None
            rows.append(dict(key=job['key'], status='valid' if obj else 'invalid_answer', parsed=obj,
                answer=final['text'], thought=trace['text'], thinking_forced=trace.get('stop_reason') != '</think>',
                thinking_finish=trace['finish_reason'], answer_finish=final['finish_reason'],
                prompt_tokens=len(b), total_input_tokens=len(b)+len(af),
                think_tokens=len(trace.get('token_ids') or tok.encode(trace['text'], add_special_tokens=False)),
                answer_tokens=len(final.get('token_ids') or tok.encode(final['text'], add_special_tokens=False)),
                run_id=os.environ.get('ORX_RUN_ID'), commit=os.environ.get('ORX_COMMIT')))
        return rows, dict(seconds=time.monotonic()-started, cache_hits=int(hit1)+int(hit2), keys=[j['key'] for j in jobs])


def prepare_maven(root, old, out):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as parquet
    import yaml
    from eci.data import doc_from_row
    paths, loaded = {}, {}
    for split in ('train', 'test'):
        path = Path(hf_hub_download(MAVEN_REPO, f'data/{split}-00000-of-00001.parquet',
                    repo_type='dataset', revision=MAVEN_REVISION))
        paths[split] = dict(path=str(path), sha256=c.sha(path))
        docs = [doc_from_row(row, i, text_field='text', ann_field='annots',
                    valid_labels={'causes', 'causedby', 'norel'}, binary_undirected=False,
                    shuffle_pair_list=False) for i, row in enumerate(parquet.read_table(path).to_pylist())]
        loaded[split] = docs
    train, test = loaded['train'], loaded['test']
    assert len(test) == 710, 'benchmark revision split size changed'
    eligible = sorted(test, key=lambda doc: (len(doc.tokens), doc.id))
    chosen = []
    import random
    rng = random.Random(v.SEED)
    for k in range(4):
        chosen.extend(rng.sample(eligible[k*len(eligible)//4:(k+1)*len(eligible)//4], 3))
    test_ids = {doc.id for doc in test}
    # Training labels can form demos; never select evaluation documents or texts as demos.
    text_hashes = {api.digest(doc.text) for doc in test}
    bank = sorted([doc for doc in train if doc.id not in test_ids and api.digest(doc.text) not in text_hashes
                   and len(doc.tokens) <= 400 and any(label != 'norel' for _, label, _ in doc.gold)],
                  key=lambda doc: api.digest([v.VERSION, 'demo', doc.id]))[:5]
    assert len(bank) == 5
    prompt_path = Path('prompts/maven_ere_standard_eci_lean_coref.yaml')
    system = yaml.safe_load(prompt_path.read_text())['system'].split('If there are no mentions')[0].strip()
    rules = {'MAVEN': system}
    documents = {}
    for doc in chosen:
        universe = set(doc.pair_list)
        positive = {(x, y) for x, lab, y in doc.gold if lab != 'norel'}
        # Preserve presence symmetry explicitly, while keeping the fixed candidate universe.
        positive |= {(y, x) for x, y in positive}
        assert positive <= universe
        documents[doc.id] = dict(id=doc.id, text=doc.text, tokens=doc.tokens, pair_list=doc.pair_list,
            gold=sorted(f'{x},{y}' for x, y in positive), sentence_of=doc.sentence_of)
    selected = {'MAVEN': [doc.id for doc in chosen]}
    docs = {'MAVEN': documents}
    instruction = (system+'\n\nReturn only a JSON object with a relations array of objects containing '
        'pair (two mention IDs separated by comma) and label (causes or causedby). '
        'Omit norel. Emit both orientations. Do not output prose or extra keys.')
    jobs = []
    for doc in chosen:
        for pass_id, demo in enumerate(bank):
            answer = {'relations': [dict(pair=f'{x},{y}', label=lab) for x, lab, y in demo.gold if lab != 'norel']}
            messages = [dict(role='system', content=instruction), dict(role='user', content='Document:\n'+demo.text),
                        dict(role='assistant', content=api.canonical(answer)),
                        dict(role='user', content='Document:\n'+doc.text)]
            jobs.append(ap.job(messages, v.document_schema(), 8192, 2048,
                                dict(dataset='MAVEN', doc_id=doc.id, pass_id=pass_id, demo_id=demo.id)))
    manifest = dict(protocol=v.VERSION, selected=selected, dataset_repo=MAVEN_REPO, revision=MAVEN_REVISION,
        source_files=paths, prompt_sha256=c.sha(prompt_path), demo_ids=[d.id for d in bank],
        bank_sha256=api.digest([dict(id=d.id, text=d.text, gold=d.gold) for d in bank]),
        requests_sha256=api.digest(jobs), policies=list(v.POLICIES), baseline='new Qwen API five greedy document reads, five different train demos, 8192 thinking + 2048 structured final',
        selection='12 documents, 3 per token-length quartile, seeded, no gold-based evaluation selection',
        scope='MAVEN test pilot; exposure to prior project runs unaudited, not claimed held out',
        no_training=True, no_weight_edit=True, local_only=True)
    ap.freeze(out/'manifest.json', manifest)
    ap.freeze(out/'documents.json', docs)
    ap.freeze(root/'AP0/api_contract.json', ap.read(old/'AP0/api_contract.json'))
    print('VC_MAVEN_PLAN', api.canonical(manifest), flush=True)
    return selected, docs, rules, jobs


def run_maven(root, old, out, deadline):
    selected, docs, rules, document_jobs = prepare_maven(root, old, out)
    doc_engine = DocumentEngine(root, deadline, namespace=v.VERSION+'-MAVEN-document')
    preflight(doc_engine, document_jobs)
    smoke = [ap.job([dict(role='user', content='Return {"relations":[]} exactly.')], v.document_schema(), 64, 128,
                     dict(validation_smoke='MAVEN_document'))]
    assert all(r['status'] == 'valid' for r in doc_engine.execute(smoke, out/'document_smoke').values())
    outputs = doc_engine.execute(document_jobs, out/'readings')
    base, quality = v.document_vote(selected, docs, document_jobs, outputs)
    ap.dump_predictions(out/'baseline.json', base)
    print('VC_MAVEN_BASELINE', api.canonical(dict(metrics=p.metrics(base, base, docs), quality=quality)), flush=True)
    engine = ap.Engine(root, deadline, namespace=v.VERSION+'-MAVEN')
    jobs = v.screen_jobs(selected, v.public_docs(docs), rules)
    preflight(engine, jobs)
    screen = engine.execute(jobs, out/'screen')
    views = v.views_from(jobs, screen)
    routes = v.routes_for(selected, v.public_docs(docs), base, views)
    ap.freeze(out/'routes.json', routes)
    rows, final_jobs = repair_jobs(routes, docs, rules)
    preflight(engine, final_jobs)
    finals = engine.execute(final_jobs, out/'repair/final')
    decisions, judgments = repair_decisions(rows, final_jobs, finals)
    a.write_jsonl(out/'repair/judgments.jsonl', judgments)
    systems = v.combine(base, views, decisions)
    metrics, _ = evaluate(systems, base, docs, selected, [])
    for name, predictions in systems.items(): ap.dump_predictions(out/f'{name}.json', predictions)
    finish(out, dict(dataset='MAVEN', selected=selected, policies=metrics, baseline_quality=quality,
        manifest_sha256=c.sha(out/'manifest.json'), scope='new local-Qwen MAVEN12 pilot; no claim against unmatched historical external-model baselines'))


def attempt_deadline(out, dataset):
    """Charge actual attempts, preserving the ceiling across repaired runs.

    A terminal failure followed by hours without an active process consumes no
    extra execution budget. All previous attempt wall records remain immutable.
    """
    path = out/'budget.json'
    now = time.time()
    state = ap.read(path) if path.exists() else dict(start=now, max_hours=HOURS[dataset])
    run = os.environ.get('ORX_RUN_ID', 'test')
    attempts = state.setdefault('attempts', {})
    spent = sum(ap.read(p)['seconds'] for p in out.glob('wall-*.json') if p.name != f'wall-{run}.json')
    for key, value in attempts.items():
        if key != run and not (out/f'wall-{key}.json').exists():
            spent += max(0, min(now, value['deadline'])-value['start'])
    if run not in attempts:
        attempts[run] = dict(start=now, deadline=now+max(0, HOURS[dataset]*3600-spent),
                             previous_active_seconds=spent)
    state['accounting'] = 'sum active attempt wall times; excludes idle time after terminal failure and queue waits'
    ap.save(path, state)
    print('VC_ATTEMPT_BUDGET', api.canonical(attempts[run]), flush=True)
    return attempts[run]['deadline']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=tuple(HOURS), required=True)
    args = parser.parse_args()
    dataset = args.dataset
    work = Path(os.environ['WORK'])
    root = work/'out/VC'/os.environ['VC_GROUP']
    old = work/'out/AP'/os.environ['AP_GROUP']
    c2 = work/'out/C2'/os.environ['C2_RUN']
    out = root/dataset
    out.mkdir(parents=True, exist_ok=True)
    print('VC_CONFIG', api.canonical(dict(dataset=dataset, root=str(root), protocol=v.VERSION,
        run=os.environ.get('ORX_RUN_ID'), commit=os.environ.get('ORX_COMMIT'), max_hours=HOURS[dataset],
        local_api=os.environ['API_BASE_URL'], policies=v.POLICIES)), flush=True)
    if (out/'terminal.json').exists() and ap.read(out/'terminal.json')['status'] == 'complete':
        print('ALREADY_COMPLETE', str(out), flush=True)
        return
    predecessor = {'ESC': 'CTB', 'MAVEN': 'ESC'}.get(dataset)
    if predecessor:
        # Wait outside API lease. Failure does not silently cancel another corpus.
        ap.wait_for(root/predecessor/'terminal.json',
                    lambda r: r.get('status') in ('complete', 'failed', 'budget_exhausted'), hours=48)
    with ap.lock(work/'ap-api.lock'):
        deadline = attempt_deadline(out, dataset)
        ap.save(out/'terminal.json', dict(status='running', run_id=os.environ.get('ORX_RUN_ID')))
        started = time.time()
        try:
            ap.check_budget(deadline)
            if dataset == 'MAVEN': run_maven(root, old, out, deadline)
            else: run_cached(dataset, root, old, c2, work, out, deadline)
        except ap.BudgetStop as exc:
            ap.save(out/'terminal.json', dict(status='budget_exhausted', reason=str(exc), cost=incremental_cost(out)))
            print('VC_BUDGET_STOP incomplete; checkpoints retained, no full-panel F1', str(exc), flush=True)
            raise
        except Exception as exc:
            ap.save(out/'terminal.json', dict(status='failed', error_type=type(exc).__name__, reason=str(exc)))
            raise
        finally:
            ap.save(out/f"wall-{os.environ.get('ORX_RUN_ID', 'test')}.json",
                    dict(start=started, end=time.time(), seconds=time.time()-started))


if __name__ == '__main__':
    main()
