"""Four-document projections of the interrupted ESC/MAVEN combination pilots.

Reuse exact frozen requests and observations, never regenerate demonstrations or
select documents by a score. The original full-corpus bundles remain untouched.
"""
from __future__ import annotations

from collections import Counter
import copy
import json
import os
from pathlib import Path
import shutil

from steering import wsp_combo as c

VERSION = 'wsp-short-20261112-v1'
PARENTS = {
    'ESC': 'e2986f95e345405886d8f1abd01474e1ff54018d58018197e1dee1d808c376fc',
    'MAVEN': '5bd120c98a5933190a0f0282da4b9328034f5de512c915ef571b4b1800ecad46',
}
CACHE_SHA256 = {
    'ESC': '1495ae268fc2cb586ed19725731e6a508ecd720ca36ac438c5e47bd037cff4b3',
    'MAVEN': None,
}
PER_DOCUMENT_FIELDS = ('predictions', 'support', 'valid_readings', 'emitted_tokens', 'invalid')


def select_ids(dataset, ids, subsets):
    """Selection sees identifiers and pre-existing cohort membership only."""
    if dataset == 'ESC':
        topics = {}
        for d in sorted(ids):
            topics.setdefault(d.split('_', 1)[0], d)
        selected = list(topics.values())[:4]
        groups = {}
        rule = 'lexicographic first document in each of first four distinct topics'
    elif dataset == 'MAVEN':
        groups = {label.replace('12', '2'): sorted(subsets[label])[:2]
                  for label in ('original12', 'extension12')}
        selected = sorted(d for values in groups.values() for d in values)
        rule = 'lexicographic first two original12 and first two extension12 identifiers'
    else:
        raise ValueError('short pilot is restricted to ESC and MAVEN')
    if len(selected) != 4 or len(set(selected)) != 4 or not set(selected) <= set(ids):
        raise ValueError('short pilot needs four distinct source documents')
    return selected, groups, rule


def project(parent, docs, prior, jobs):
    ids, subsets, rule = select_ids(parent['dataset'], parent['documents'], parent['subsets'])
    selected = set(ids)
    documents = {d: docs[d] for d in sorted(selected)}
    requests = [j for j in jobs if j['meta']['doc_id'] in selected]
    if Counter(j['meta']['doc_id'] for j in requests) != Counter({d: c.READS for d in selected}):
        raise ValueError('five exact requests per selected document required')
    for j in requests:
        if j['key'] != c.digest({k: v for k, v in j.items() if k != 'key'}):
            raise ValueError('source request fingerprint mismatch')
    snapshot = copy.deepcopy(prior)
    for field in PER_DOCUMENT_FIELDS:
        snapshot[field] = {d: value for d, value in prior[field].items() if d in selected}
    return documents, snapshot, requests, subsets, rule


def import_cache(source, out, parent, plan, source_jobs, jobs):
    """Keep raw source rows separately; rebind only the manifest of imported copies."""
    source_path = source / 'readings.jsonl'
    expected_sha = CACHE_SHA256[parent['dataset']]
    actual_sha = c.sha(source_path) if source_path.exists() else None
    if actual_sha != expected_sha:
        raise ValueError('source cache changed since cancellation audit')
    rows = c.read_cache(source_path, source_jobs, parent, partial=True)
    keys = {j['key'] for j in jobs}
    originals = [r for (arm, key), r in rows.items() if key in keys]
    c.freeze(out / 'imported_source_rows.json', originals)
    imported = [dict(r, manifest_id=plan['id'], imported_from_manifest=parent['id']) for r in originals]
    c.freeze(out / 'reuse.json', dict(source_manifest=parent['id'], source_cache_sha256=actual_sha,
        source_rows_sha256=c.digest(originals), rows=len(imported),
        tokens=sum(r['num_tokens'] for r in originals),
        note='Answer, thought, validity, tokens, run and commit preserved. All reused tokens count in evaluation cost.'))
    path = out / 'readings.jsonl'
    if path.exists():
        current = c.read_cache(path, jobs, plan, partial=True)
        if any(current.get((r['arm'], r['key'])) != r for r in imported):
            raise ValueError('imported observation changed or disappeared')
    else:
        temporary = out / 'readings.import.tmp'
        temporary.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in imported))
        temporary.replace(path)
    print('WSP_SHORT_REUSE ' + json.dumps(dict(dataset=parent['dataset'],
          imported=len(imported), scheduled=2*len(jobs), new=2*len(jobs)-len(imported))), flush=True)


def prepare(args):
    source = Path(os.environ['WSP_SHORT_SOURCE'])
    out = Path(args.out)
    if source.resolve() == out.resolve():
        raise ValueError('never overwrite the full pilot')
    parent, all_docs, all_prior, all_jobs = c.load_bundle(source)
    if parent['id'] != PARENTS.get(args.dataset) or parent['dataset'] != args.dataset:
        raise ValueError('unexpected parent bundle')
    if not 0 < args.max_hours <= 2:
        raise ValueError('short pilot active budget must be at most two hours')
    expected_engine = dict(model=args.model, max_model_len=args.max_model_len, batch=args.batch,
                           gpu_memory_utilization=args.gpu_memory_utilization, vllm='0.32.0')
    if parent['engine'] != expected_engine:
        raise ValueError('engine drift prevents matched cache reuse')
    for name, fingerprint in parent['implementation_sha256'].items():
        if c.sha(c.HERE / name) != fingerprint:
            raise ValueError(f'original implementation changed: {name}')
    if c.sha(source / 'directions.npz') != parent['directions_sha256']:
        raise ValueError('source directions changed')
    docs, prior, jobs, subsets, rule = project(parent, all_docs, all_prior, all_jobs)
    plan = copy.deepcopy(parent)
    plan.pop('id')
    plan.update(version=VERSION, parent_manifest=parent['id'], parent_documents=len(all_docs),
        parent_historical_replay=parent['historical_replay'], documents=sorted(docs), subsets=subsets,
        selection=dict(rule=rule, uses_gold=False, uses_scores=False, documents_per_dataset=4),
        max_active_hours=args.max_hours, reading_jobs_sha256=c.digest(jobs), documents_sha256=c.digest(docs),
        historical_snapshot_sha256=c.digest(prior),
        historical_replay=c.metrics({d: c.counts(c.symmetric(set(map(tuple, ps))), c.gold_pairs(args.dataset, docs[d]))
                                    for d, ps in prior['predictions'].items()}),
        reuse_source_cache_sha256=CACHE_SHA256[args.dataset],
        scope='Exploratory convenience pilot on four reused-development documents; '
              'four ESC topics or two original plus two extension MAVEN documents. '
              'Unadjusted intervals on four units are descriptive only. No full-benchmark record claim.',
        active_budget_note='New user-requested reduced scope; previous full-pilot runtime remains in its archived ledger. '
                           'Two-hour cumulative new active limit checked between batches; one batch may overrun.')
    plan['implementation_sha256'].update({n: c.sha(c.HERE / n) for n in ('wsp_short.py', 'run_wsp_combo.sh')})
    plan['id'] = c.digest(plan)
    out.mkdir(parents=True, exist_ok=True)
    for name, value in [('manifest.json', plan), ('documents.json', docs), ('historical.json', prior),
                        ('reading_jobs.json', jobs), ('rules.json', json.loads((source / 'rules.json').read_text())),
                        ('model_identity.json', json.loads((source / 'model_identity.json').read_text()))]:
        c.freeze(out / name, value)
    if not (out / 'directions.npz').exists():
        shutil.copyfile(source / 'directions.npz', out / 'directions.npz')
    if c.sha(out / 'directions.npz') != plan['directions_sha256']:
        raise ValueError('copied direction drift')
    import_cache(source, out, parent, plan, all_jobs, jobs)
    print('WSP_SHORT_PLAN ' + json.dumps(plan), flush=True)
    return plan


if __name__ == '__main__':
    c.prepare = prepare
    c.main()
