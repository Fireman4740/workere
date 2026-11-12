"""50-document follow-ups to W5/W6 using frozen data, directions and prompts."""
from __future__ import annotations

import copy
import gzip
import json
import os
from pathlib import Path
import shutil

from steering import wsp_combo as c

VERSION = 'wsp-scale50-20261112-v1'
PARENTS = {'ESC': 'e2986f95e345405886d8f1abd01474e1ff54018d58018197e1dee1d808c376fc',
           'MAVEN': '5bd120c98a5933190a0f0282da4b9328034f5de512c915ef571b4b1800ecad46'}
ESC_SHORT = '546b70beafaeffa0d5864b97f09b6b38d03bcb574ef8a1f9183479af429a5d8c'
ESC_CACHE_HASHES = ('1495ae268fc2cb586ed19725731e6a508ecd720ca36ac438c5e47bd037cff4b3',
                    'f66b860b395e3a00075cce5a61d3edf508708acfef68d37076ee1ef051052b3c')


def maven_pair_schema():
    schema = copy.deepcopy(c.document_schema())
    # The source corpus uses e<number> mention IDs. Require two IDs in each
    # pair: W6's unconstrained schema accepted an isolated ID as a string.
    schema['items']['properties']['pair']['pattern'] = r'^e[0-9]+,e[0-9]+$'
    return schema


def maven_jobs(docs, original_docs, original_jobs):
    """Rebuild the five frozen demos without including evaluation labels."""
    demos = {}
    for j in original_jobs:
        template = j['messages'][:3]
        key = j['meta']['demo_id']
        if key in demos and demos[key] != template:
            raise ValueError('demonstration template drift')
        demos[key] = template
    if len(demos) != c.READS:
        raise ValueError('expected five frozen MAVEN demonstrations')
    schema = maven_pair_schema()
    result, old = [], []
    for doc_id, doc in sorted(docs.items()):
        ids = sorted(demos, key=lambda d: c.digest([c.VERSION, 'MAVEN', doc_id, d]))[:c.READS]
        for pass_id, demo_id in enumerate(ids):
            messages = copy.deepcopy(demos[demo_id]) + [dict(role='user', content='Document:\n' + doc['text'] +
                '\nAfter thinking, return only the JSON array of pair/label objects; omit norel.')]
            meta = dict(dataset='MAVEN', doc_id=doc_id, pass_id=pass_id, demo_id=demo_id, kind='reading')
            result.append(c.job(meta, messages, schema, 8192, 4096))
            if doc_id in original_docs:
                old.append(c.job(meta, messages, c.document_schema(), 8192, 4096))
    if old != original_jobs:
        raise ValueError('MAVEN original prompts could not be reproduced exactly')
    return result


def load_sources(args):
    root = Path(os.environ['WSP_FULL_BUNDLE'])
    source = root/args.dataset
    parent, original_docs, prior, original_jobs = c.load_bundle(source)
    if parent['id'] != PARENTS.get(args.dataset):
        raise ValueError('unexpected source bundle')
    expected_engine = dict(model=args.model, max_model_len=args.max_model_len, batch=args.batch,
                           gpu_memory_utilization=args.gpu_memory_utilization, vllm='0.32.0')
    if parent['engine'] != expected_engine:
        raise ValueError('engine drift')
    for name, fingerprint in parent['implementation_sha256'].items():
        if c.sha(c.HERE/name) != fingerprint:
            raise ValueError(f'original implementation changed: {name}')
    if c.sha(source/'directions.npz') != parent['directions_sha256']:
        raise ValueError('source direction drift')
    payload = json.loads(gzip.decompress((c.HERE/'scale50_inputs.json.gz').read_bytes()))
    data = payload[args.dataset.lower()]
    if data['parent_manifest'] != parent['id'] or payload['selection_uses_gold'] is not False:
        raise ValueError('incorrect input projection')
    docs = dict(original_docs)
    for d in data['extra']:
        if d['id'] in docs:
            raise ValueError('duplicate evaluation document')
        docs[d['id']] = d
    if len(docs) != 50:
        raise ValueError('exactly 50 documents required')
    for d in docs.values():
        c.gold_pairs(args.dataset, d)  # check corpus consistency only, never used in prompts
    if args.dataset == 'ESC':
        prompt = json.loads((c.HERE/'data/esc/prompt.json').read_text())
        if any(d['id'] in {q['id'] for q in prompt['demos']} for d in data['extra']):
            raise ValueError('evaluation/demonstration overlap')
        jobs = c.make_reading_jobs('ESC', {d:c.public(x) for d,x in docs.items()}, prompt)
        if [j for j in jobs if j['meta']['doc_id'] in original_docs] != original_jobs:
            raise ValueError('ESC previous prompts changed; cache reuse forbidden')
        subsets = dict(historical40=sorted(original_docs), extension10=sorted(d['id'] for d in data['extra']))
    else:
        jobs = maven_jobs(docs, original_docs, original_jobs)
        original12 = sorted(parent['subsets']['original12'])
        subsets = dict(original12=original12, other38=sorted(set(docs)-set(original12)))
    if len(jobs) != 250 or len({j['key'] for j in jobs}) != 250:
        raise ValueError('need 250 unique matched reading prompts')
    return source, parent, docs, prior, jobs, subsets, data, original_jobs


def import_esc_cache(out, source, parent, short_source, jobs, original_jobs, plan):
    paths = [source/'readings.jsonl', short_source/'readings.jsonl']
    if tuple(c.sha(p) for p in paths) != ESC_CACHE_HASHES:
        raise ValueError('ESC source observation cache changed')
    short_plan, _, _, short_jobs = c.load_bundle(short_source)
    if short_plan['id'] != ESC_SHORT:
        raise ValueError('unexpected completed short pilot')
    sources = [c.read_cache(paths[0], original_jobs, parent, partial=True),
               c.read_cache(paths[1], short_jobs, short_plan)]
    allowed = {j['key'] for j in jobs}
    canonical = {}
    originals = []
    for provenance, rows in zip((parent['id'], short_plan['id']), sources):
        for key, row in rows.items():
            if key[1] not in allowed:
                raise ValueError('source request missing from 50-document plan')
            originals.append(dict(source_manifest=provenance, observation=row))
            normalized = {k:v for k,v in row.items() if k not in ('manifest_id','imported_from_manifest')}
            if key in canonical and canonical[key] != normalized:
                raise ValueError('conflicting duplicate cached reading')
            canonical[key] = normalized
    imported = [dict(row, manifest_id=plan['id'], imported_from_manifest='ESC prior cache')
                for _, row in sorted(canonical.items())]
    if len(imported) != 53:
        raise ValueError(f'unexpected reuse count {len(imported)}')
    c.freeze(out/'imported_source_rows.json', originals)
    c.freeze(out/'reuse.json', dict(source_manifests=[parent['id'], short_plan['id']],
        source_cache_sha256=list(ESC_CACHE_HASHES), observations=len(originals), unique=len(imported),
        emitted_tokens=sum(r['num_tokens'] for r in imported),
        note='Raw source rows preserved; observations and costs unchanged; verification is rerun.'))
    path = out/'readings.jsonl'
    if path.exists():
        current = c.read_cache(path, jobs, plan, partial=True)
        if any(current.get((r['arm'],r['key'])) != r for r in imported):
            raise ValueError('imported observation changed or disappeared')
    else:
        tmp = out/'readings.import.tmp'
        tmp.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in imported))
        tmp.replace(path)
    print('WSP_SCALE50_REUSE '+json.dumps(dict(imported=len(imported), total=2*len(jobs))),flush=True)


def prepare(args):
    out = Path(args.out)
    source, parent, docs, prior, jobs, subsets, data, old_jobs = load_sources(args)
    if not 0 < args.max_hours <= 36:
        raise ValueError('active budget must be within 36 hours')
    if out.resolve() == source.resolve():
        raise ValueError('cannot overwrite frozen source')
    plan = copy.deepcopy(parent)
    plan.pop('id')
    plan.update(version=VERSION, parent_manifest=parent['id'], documents=sorted(docs), subsets=subsets,
        parent_documents=len(parent['documents']), selection_rule=data['selection'], selection_uses_gold=False,
        additional_source_sha256=data['source_sha256'], max_active_hours=args.max_hours,
        documents_sha256=c.digest(docs), reading_jobs_sha256=c.digest(jobs),
        historical_scope='Only the historical40 ESC or original12 MAVEN subset has a historical reference; '
                         '50-document comparisons use matched new base5.',
        historical_augmentation='Historical comparison is reported on its matching subset; '
                                'vote-family/verification routing for all 50 uses new readings.',
        pair_format='MAVEN pair schema constrained to e<number>,e<number> after W6 singleton-format failures' if args.dataset=='MAVEN'
                    else 'unchanged from W5',
        scope='50-document user-requested expansion of previously studied corpora; '
              'exploratory variants and no independent confirmation; evaluation subsets selected without gold.')
    plan['implementation_sha256'].update({n:c.sha(c.HERE/n) for n in
        ('wsp_scale50.py','scale50_select.py','scale50_inputs.json.gz','run_wsp_combo.sh')})
    plan['id'] = c.digest(plan)
    out.mkdir(parents=True, exist_ok=True)
    for name,value in [('manifest.json',plan),('documents.json',docs),('historical.json',prior),
                       ('reading_jobs.json',jobs),('rules.json',json.loads((source/'rules.json').read_text())),
                       ('model_identity.json',json.loads((source/'model_identity.json').read_text()))]:
        c.freeze(out/name,value)
    if not (out/'directions.npz').exists():
        shutil.copyfile(source/'directions.npz',out/'directions.npz')
    if c.sha(out/'directions.npz') != plan['directions_sha256']:
        raise ValueError('copied directions drift')
    if args.dataset == 'ESC':
        import_esc_cache(out,source,parent,Path(os.environ['WSP_SHORT_BUNDLE']),jobs,old_jobs,plan)
    print('WSP_SCALE50_PLAN '+json.dumps(plan),flush=True)
    return plan


if __name__ == '__main__':
    c.prepare = prepare
    c.main()
