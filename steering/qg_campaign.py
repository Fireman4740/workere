"""Tracked QG1/QG2 API experiment. Reuses AP4, never edits its outputs."""
import argparse
from collections import Counter
import os
from pathlib import Path
import time

from steering import ap_campaign as ap
from steering import ap_protocol as p
from steering import campaign_adjudicate as a
from steering import campaign_api as api
from steering import campaign_blocks as c
from steering import campaign_facts as f
from steering import qg_protocol as q


def prepare(old, c2, work, root):
    previous = ap.read(old/'AP4/report.json')
    assert previous['status'] == 'complete'
    ctb, reference, esc, rules = c.inputs()
    docs = dict(CTB=ctb, ESC=esc)
    selected = dict(CTB=ap.read(old/'AP0/report.json')['selected']['CTB'],
                    ESC=previous['selected']['ESC'])
    saved = ap.read(old/'AP4/predictions.json')
    base = {ds: {d: set(map(tuple, saved[ds][d])) for d in ids} for ds, ids in selected.items()}
    replay = p.metrics(base, base, docs)
    for ds, ids in selected.items():
        assert replay[ds]['per_document'] == {d: previous['metrics'][ds]['per_document'][d] for d in ids}, \
            'retained AP4 predictions fail exact per-document replay'
    print('QG_REFERENCE_REPLAY', api.canonical({ds: {k: r[k] for k in ('documents', 'tp', 'fp', 'fn', 'f1')}
                                               for ds, r in replay.items()}), flush=True)
    _, history = f.historical_baselines(selected, ctb, reference, esc, work)
    historical = {ds: {d: a.unordered(history[ds]['n1_p3' if ds == 'CTB' else 'qwen_p2_sym'][d])
                       for d in ids} for ds, ids in selected.items()}
    _, _, views = a.bundle(c2)
    screen = old/'AP4/screen'
    requests = c.jsonl(screen/'requests.jsonl')
    rows = {r['key']: r for r in c.jsonl(screen/'responses.jsonl')}
    assert set(rows) == {j['key'] for j in requests}
    for j in requests:
        meta, row = j['meta'], rows[j['key']]
        if row['status'] != 'valid':
            continue
        target = views.setdefault((meta['dataset'], meta['doc_id'], meta['order']), {})
        for i, pair in enumerate(meta['pairs']):
            pair = tuple(sorted(pair))
            assert pair not in target, 'overlapping initial and expansion screens'
            target[pair] = row['parsed'][str(i)] != 'N'
    systems = q.candidate_systems(selected, base, historical, views)
    # The generation planner receives source and candidates, with gold/coreference removed.
    public = {ds: {d: {k: docs[ds][d][k] for k in ('id', 'text', 'pair_list')} for d in ids}
              for ds, ids in selected.items()}
    blocks = q.blocks(selected, public, systems)
    sources = [old/'AP4/report.json', old/'AP4/predictions.json', old/'AP0/api_contract.json',
               old/'AP0/report.json', screen/'requests.jsonl', screen/'responses.jsonl',
               c2/'requests.jsonl', c2/'generations.jsonl',
               work/'out/N1/gen.jsonl', work/'out/P3/gen.jsonl', work/'out/P2/gen.jsonl']
    manifest = dict(protocol=q.VERSION, selected=selected, block_size=q.BLOCK_SIZE,
                    budgets=dict(thinking=q.THINKING, answer=q.ANSWER),
                    source_hashes={str(path): c.sha(path) for path in sources},
                    data_hashes={str(path.relative_to(c.DATA)): c.sha(path) for path in c.DATA.rglob('*.json')},
                    requests_per_arm=len(blocks)*2, blocks_per_dataset=dict(Counter(b['dataset'] for b in blocks)),
                    pairs_per_dataset={ds: sum(len(b['pairs']) for b in blocks if b['dataset'] == ds) for ds in selected},
                    scope='CTB20/ESC20 reused development, binary causal presence, no direction claim',
                    local_only=True, no_training=True, no_weight_edit=True)
    ap.freeze(root/'manifest.json', manifest)
    ap.freeze(root/'blocks.json', blocks)
    ap.freeze(root/'AP0/api_contract.json', ap.read(old/'AP0/api_contract.json'))
    ap.dump_predictions(root/'reference.json', base)
    ap.dump_predictions(root/'historical.json', historical)
    print('QG_PLAN', api.canonical(manifest), flush=True)
    return selected, docs, rules, base, historical, systems, blocks


def jobs_for(blocks, rules, method):
    jobs = []
    for block in blocks:
        for order in (0, 1):
            job = ap.job(q.messages(block, order, method, rules[block['dataset']]), q.spec(block, method),
                         q.THINKING, q.ANSWER,
                         dict(protocol=q.VERSION, block=block['key'], dataset=block['dataset'],
                              doc_id=block['doc_id'], method=method, order=order))
            jobs.append(job)
    # Pair both presentations in a deterministic schedule, identical across arms.
    return sorted(jobs, key=lambda j: api.digest([q.VERSION, j['meta']['block'], j['meta']['order']]))


def preflight(engine, jobs):
    lengths = []
    for job in jobs:
        text = engine.tokenizer.apply_chat_template(job['messages'], tokenize=False,
                    add_generation_prompt=True, enable_thinking=True, preserve_thinking=True)
        length = len(engine.tokenizer.encode(text, add_special_tokens=False))
        # Conservative safety margin for the closed thinking tag and tokenization boundary.
        assert length+job['thinking']+32 <= 16384, 'full panel must fit, no silent exclusions'
        assert length+job['thinking']+job['answer_tokens']+32 <= 32768
        lengths.append(length)
    print('QG_PREFLIGHT', api.canonical(dict(requests=len(jobs), max_input=max(lengths, default=0),
                                           input_tokens=sum(lengths))), flush=True)


def smoke(engine, out, method):
    block = dict(pairs=[['a', 'b']], options=[[False], [True]])
    jobs = [ap.job([{'role':'user', 'content':'Return only the requested JSON. Use G0 for choice, N for pair 0.'}],
                   q.spec(block, method), 64, 128, dict(protocol=q.VERSION, smoke=method))]
    rows = engine.execute(jobs, out/'smoke')
    assert all(r['status'] == 'valid' for r in rows.values()), 'new QG schema smoke failed'
    print('QG_SMOKE passed', method, flush=True)


def evaluate(blocks, jobs, outputs, method, base, historical, systems, docs):
    indexed = {(j['meta']['block'], j['meta']['order']): outputs[j['key']] for j in jobs}
    predictions, single, quality = q.combine(base, blocks, indexed, method)
    metrics = p.metrics(predictions, base, docs)
    union = {ds: {d: set().union(*(s[ds][d] for s in systems.values())) for d in values}
             for ds, values in base.items()}
    # Post-run diagnostic only; labels never determine routes/options or a threshold.
    report = dict(method=method, metrics=metrics, versus_historical=p.metrics(predictions, historical, docs),
                  single_presentations={str(i): p.metrics(s, base, docs) for i, s in enumerate(single)},
                  agreement=quality, candidate_union=p.metrics(union, base, docs),
                  candidate_union_note='Recall coverage diagnostic; union is not the selected output or an attainable F1 bound.')
    return predictions, report


def repeat_audit(root, out, stop, method, blocks, jobs, outputs, base):
    """Fresh HTTP cache namespace; exact prompts repeated on a frozen small panel."""
    chosen = []
    for ds in base:
        chosen.extend(sorted((b for b in blocks if b['dataset'] == ds),
                             key=lambda b: api.digest([q.VERSION, 'repeat', b['key']]))[:8])
    keys = {b['key'] for b in chosen}
    repeated_jobs = [j for j in jobs if j['meta']['block'] in keys]
    engine = ap.Engine(root, stop, namespace=q.VERSION+'-'+method+'-repeat')
    fresh = engine.execute(repeated_jobs, out/'repeat')
    old_rows = {(j['meta']['block'], j['meta']['order']): outputs[j['key']] for j in repeated_jobs}
    new_rows = {(j['meta']['block'], j['meta']['order']): fresh[j['key']] for j in repeated_jobs}
    old, _, _ = q.combine(base, chosen, old_rows, method)
    new, _, _ = q.combine(base, chosen, new_rows, method)
    return {ds: dict(blocks=sum(b['dataset'] == ds for b in chosen),
                     pairs=sum(len(b['pairs']) for b in chosen if b['dataset'] == ds),
                     changed_binary_decisions=sum(len(old[ds][d] ^ new[ds][d]) for d in base[ds]),
                     scope='Exact prompts, fresh responses, smaller batches; technical repeatability, not independent validation')
            for ds in base}


def deadline(root, method):
    path = root/'budget.json'
    state = ap.read(path) if path.exists() else dict(start=time.time(), stages={})
    state['stages'].setdefault(method, time.time())
    ap.save(path, state)
    return min(state['start']+12*3600, state['stages'][method]+6*3600)


def run(root, old, c2, work, method):
    out = root/method
    out.mkdir(parents=True, exist_ok=True)
    if (out/'terminal.json').exists() and ap.read(out/'terminal.json')['status'] == 'complete':
        print('ALREADY_COMPLETE', str(out), flush=True)
        return
    with ap.lock(work/'ap-api.lock'):
        stop = deadline(root, method)
        started = time.time()
        try:
            ap.check_budget(stop)
            selected, docs, rules, base, historical, systems, blocks = prepare(old, c2, work, root)
            jobs = jobs_for(blocks, rules, method)
            # Shared executor reads only the persistent server identity contract.
            engine = ap.Engine(root, stop, namespace=q.VERSION+'-'+method)
            preflight(engine, jobs)
            smoke(engine, out, method)
            outputs = engine.execute(jobs, out/'graphs')
            assert set(outputs) == {j['key'] for j in jobs}
            predictions, report = evaluate(blocks, jobs, outputs, method, base, historical, systems, docs)
            ap.dump_predictions(out/'predictions.json', predictions)
            report['repeatability'] = repeat_audit(root, out, stop, method, blocks, jobs, outputs, base)
            other = 'edit' if method == 'select' else 'select'
            if (root/other/'terminal.json').exists() and ap.read(root/other/'terminal.json')['status'] == 'complete':
                alternate = ap.read(root/other/'predictions.json')
                alternate = {ds: {d: set(map(tuple, v)) for d, v in values.items()} for ds, values in alternate.items()}
                report['paired_vs_other_arm'] = dict(other=other, metrics=p.metrics(predictions, alternate, docs))
            report.update(selected=selected, status='complete', run_id=os.environ.get('ORX_RUN_ID'),
                          commit=os.environ.get('ORX_COMMIT'), cost=ap.costs(out),
                          scope='reused development; not independent confirmation',
                          cost_note='Incremental QG calls including schema smoke. Archived votes and C2/AP generation costs excluded.',
                          manifest_sha256=c.sha(root/'manifest.json'), blocks_sha256=c.sha(root/'blocks.json'))
            ap.save(out/'report.json', report)
            ap.save(out/'terminal.json', dict(status='complete', report_sha256=c.sha(out/'report.json')))
            print('QG_RESULT', api.canonical(report), flush=True)
        except ap.BudgetStop as exc:
            ap.save(out/'terminal.json', dict(status='budget_exhausted', reason=str(exc), cost=ap.costs(out)))
            print('QG_BUDGET_STOP incomplete; no full-panel F1', str(exc), flush=True)
        finally:
            ap.save(out/f"wall-{os.environ.get('ORX_RUN_ID', 'test')}.json",
                    dict(start=started, end=time.time(), seconds=time.time()-started))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=('select', 'edit'), required=True)
    args = parser.parse_args()
    work = Path(os.environ['WORK'])
    root = work/'out/QG'/os.environ['QG_GROUP']
    old = work/'out/AP'/os.environ['AP_GROUP']
    c2 = work/'out/C2'/os.environ['C2_RUN']
    print('QG_CONFIG', api.canonical(dict(protocol=q.VERSION, method=args.method,
        root=str(root), source=str(old), run=os.environ.get('ORX_RUN_ID'), commit=os.environ.get('ORX_COMMIT'),
        max_hours_per_arm=6, total_max_hours=12, block_size=q.BLOCK_SIZE, thinking=q.THINKING,
        answer=q.ANSWER, server=os.environ['API_BASE_URL'], model='qwen3.8-27b')), flush=True)
    run(root, old, c2, work, args.method)


if __name__ == '__main__':
    main()
