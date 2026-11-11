"""W4--W6: frozen weight edit x diverse readings x selective verification.

Exploratory development pilots, never a continuation of W2's confirmation test.
Planning takes public text/candidates only. Gold enters scoring after jobs freeze.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from steering.weight_space import LAYERS, counts, digest, install_writers, metrics
from steering.wsp_pilot import apply_arm, read_rows, sha, write_json

VERSION = 'wsp-combinations-20261111-v1'
MODEL = '/app/models/Qwen3.8-27B-W4A16-AutoRound'
MAVEN_REPO = 'Nofing/MAVEN-ERE-standard-eci-workere'
MAVEN_REVISION = '830a58ee7a092d5ca27b4a8c28c18e2c82abf814'
SEED = 20261111
READS = 5
ROUTE_CAP = 24
BLOCK = 8
VERIFY_BUDGET = 4096
POLICIES = ('base_filter', 'base_repair', 'base_add', 'cc_repair',
            'joint_repair', 'joint_add', 'strict_add_base_filter')


def freeze(path, value):
    path = Path(path)
    value = json.loads(json.dumps(value, ensure_ascii=False))
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f'frozen file changed: {path}')
    write_json(path, value)


def unique_object(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def pairs(doc):
    return {tuple(sorted(map(str, p))) for p in doc['pair_list'] if p[0] != p[1]}


def symmetric(values):
    return {p for a, b in values for p in ((a, b), (b, a))}


def public(doc):
    return {k: doc[k] for k in ('id', 'text', 'pair_list')}


def gold_pairs(dataset, doc):
    if dataset == 'CTB':
        result = {(a, b) for a, label, b in doc['gold'] if label != 'norel'}
    else:
        result = {tuple(p.split(',', 1)) for p in doc['gold']}
    if not result <= symmetric(pairs(doc)):
        raise ValueError('gold outside candidate universe')
    if result != symmetric({tuple(sorted(p)) for p in result}):
        raise ValueError('benchmark presence gold must be symmetric')
    return result


def document_schema():
    item = {'type': 'object', 'properties': {
        'pair': {'type': 'string'}, 'label': {'type': 'string', 'enum': ['causes', 'causedby', 'norel']}},
        'required': ['pair', 'label'], 'additionalProperties': False}
    return {'type': 'array', 'items': item}


def parse_document(text):
    try:
        values = json.loads(text, object_pairs_hook=unique_object)
        if not isinstance(values, list):
            return None
        positive, negative = set(), set()
        for item in values:
            if not isinstance(item, dict) or set(item) != {'pair', 'label'}:
                return None
            if not isinstance(item['pair'], str) or item['label'] not in ('causes', 'causedby', 'norel'):
                return None
            ids = [s.strip() for s in item['pair'].split(',')]
            if len(ids) != 2 or not all(ids) or ids[0] == ids[1]:
                return None
            pair = tuple(sorted(ids))
            (negative if item['label'] == 'norel' else positive).add(pair)
        return None if positive & negative else positive
    except (TypeError, ValueError):
        return None


def block_schema(n):
    return {'type': 'object', 'properties': {str(i): {'type': 'string', 'enum': ['A', 'B', 'N']}
            for i in range(n)}, 'required': [str(i) for i in range(n)], 'additionalProperties': False}


def parse_block(text, n):
    try:
        value = json.loads(text, object_pairs_hook=unique_object)
        if (not isinstance(value, dict) or set(value) != {str(i) for i in range(n)} or
                any(v not in ('A', 'B', 'N') for v in value.values())):
            return None
        return value
    except (TypeError, ValueError):
        return None


def arm(name):
    if name not in ('base', 'cc'):
        raise ValueError(name)
    return dict(name=name, family='cc' if name == 'cc' else None,
                alpha=.1 if name == 'cc' else 0., layers=list(LAYERS))


def job(meta, messages, schema, thinking, answer_tokens):
    result = dict(meta=meta, messages=messages, schema=schema,
                  thinking=thinking, answer_tokens=answer_tokens)
    result['key'] = digest(result)
    return result


def load_maven(history):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as parquet
    import yaml
    from eci.data import doc_from_row

    old = json.loads((history / 'VC/vc-20261109-combinations-v1/MAVEN/manifest.json').read_text())
    loaded, sources = {}, {}
    for split in ('train', 'test'):
        path = Path(hf_hub_download(MAVEN_REPO, f'data/{split}-00000-of-00001.parquet',
                    repo_type='dataset', revision=MAVEN_REVISION))
        if sha(path) != old['source_files'][split]['sha256']:
            raise ValueError('MAVEN revision/hash drift')
        sources[split] = sha(path)
        source_rows = parquet.read_table(path).to_pylist()
        if split == 'train':
            source_rows = [r for r in source_rows if str(r['id']) in old['demo_ids']]
        loaded[split] = [doc_from_row(row, i, text_field='text', ann_field='annots',
            valid_labels={'causes', 'causedby', 'norel'}, binary_undirected=False,
            shuffle_pair_list=False) for i, row in enumerate(source_rows)]
        del source_rows
    if len(loaded['test']) != 710:
        raise ValueError('MAVEN test size changed')
    eligible = sorted(loaded['test'], key=lambda d: (len(d.tokens), d.id))
    original = old['selected']['MAVEN']
    extension = []
    rng = random.Random(SEED)
    for k in range(4):
        bucket = [d for d in eligible[k*len(eligible)//4:(k+1)*len(eligible)//4]
                  if d.id not in original]
        extension.extend(d.id for d in rng.sample(bucket, 3))
    ids = original + extension
    bank_ids = old['demo_ids']
    train = {d.id: d for d in loaded['train']}
    bank = [train[d] for d in bank_ids]
    test_ids = {d.id for d in eligible}
    test_texts = {digest(d.text) for d in eligible}
    if set(bank_ids) & test_ids or any(digest(d.text) in test_texts for d in bank):
        raise ValueError('MAVEN demo leakage')
    docmap = {d.id: d for d in eligible}
    documents = {}
    for doc_id in ids:
        d = docmap[doc_id]
        gold = symmetric({tuple(sorted((a, b))) for a, label, b in d.gold if label != 'norel'})
        documents[doc_id] = dict(id=doc_id, text=d.text, pair_list=d.pair_list,
                                gold=sorted(','.join(p) for p in gold))
    path = HERE.parent / 'prompts/maven_ere_standard_eci_lean_coref.yaml'
    system = yaml.safe_load(path.read_text())['system'].split('If there are no mentions')[0].strip()
    prompt = dict(system=system, user_template='Document:\n{doc_text}', demos=[
        dict(id=d.id, text=d.text, answer=json.dumps([dict(pair=f'{a},{b}', label=label)
             for a, label, b in d.gold if label != 'norel'])) for d in bank])
    return documents, prompt, dict(original12=original, extension12=extension), sources


def load_inputs(dataset, history):
    if dataset == 'MAVEN':
        return load_maven(history)
    if dataset == 'CTB':
        rows = json.loads((HERE / 'data/ctb-suite.json').read_text())['documents']
        ids = json.loads((HERE / 'data/reference.json').read_text())['qwen_ids']
        prompt_path = HERE / 'data/ctb-prompt.json'
        paths = ['ctb-suite.json', 'ctb-prompt.json', 'reference.json']
    else:
        rows = json.loads((HERE / 'data/esc/test-s40.json').read_text())
        ids = sorted(d['id'] for d in rows)
        prompt_path = HERE / 'data/esc/prompt.json'
        paths = ['esc/test-s40.json', 'esc/prompt.json', 'esc/baseline.json']
    docs = {d['id']: d for d in rows if d['id'] in ids}
    expected = 47 if dataset == 'CTB' else 40
    if len(docs) != expected:
        raise ValueError('document count drift')
    return docs, json.loads(prompt_path.read_text()), {}, {p: sha(HERE / 'data' / p) for p in paths}


def make_reading_jobs(dataset, docs, prompt):
    """Same five distinct train demonstrations for both arms. No evaluation gold."""
    ids = set(docs)
    texts = {digest(d['text']) for d in docs.values()}
    bank = [d for d in prompt['demos'] if d['id'] not in ids and digest(d['text']) not in texts]
    if len(bank) < READS:
        raise ValueError('need five non-leaking demonstrations')
    jobs = []
    for doc_id, doc in sorted(docs.items()):
        eligible = [d for d in bank if dataset != 'ESC' or str(d['topic']) != doc_id.split('_', 1)[0]]
        selected = sorted(eligible, key=lambda d: digest([VERSION, dataset, doc_id, d['id']]))[:READS]
        if len(selected) != READS:
            raise ValueError('need five demonstrations outside the evaluated document/topic')
        for pass_id, demo in enumerate(selected):
            messages = [dict(role='system', content=prompt['system']),
                dict(role='user', content=prompt['user_template'].format(doc_text=demo['text'])),
                dict(role='assistant', content=demo['answer']),
                dict(role='user', content=prompt['user_template'].format(doc_text=doc['text']) +
                     '\nAfter thinking, return only the JSON array of pair/label objects; omit norel.')]
            jobs.append(job(dict(dataset=dataset, doc_id=doc_id, pass_id=pass_id, demo_id=demo['id'],
                                 kind='reading'), messages, document_schema(), 8192, 4096))
    return jobs


def vote(readings, threshold=.5):
    """All scheduled new readings count. An invalid reading contributes no edge."""
    if not readings:
        return set()
    total = Counter(p for reading in readings for p in (reading or set()))
    return {p for p, n in total.items() if n / len(readings) >= threshold}


def historical(dataset, docs, history):
    """Reproduce old scoring exactly, retain old denominators and actual token cost."""
    from eci.parse import parse_triples
    found, raw_cost, hashes, invalid = defaultdict(list), Counter(), {}, Counter()
    if dataset == 'MAVEN':
        root = history / 'VC/vc-20261109-combinations-v1/MAVEN/readings'
        requests = {r['key']: r for r in read_rows(root / 'requests.jsonl')}
        for row in read_rows(root / 'responses.jsonl'):
            meta = requests[row['key']]['meta']; d = meta['doc_id']
            if d not in docs:
                continue
            obj = json.loads(row['answer']) if row['status'] == 'valid' else None
            ps = {tuple(sorted(x.strip() for x in r['pair'].split(','))) for r in obj['relations']} if obj else set()
            found[d].append(ps & pairs(docs[d])); invalid[d] += obj is None
            raw_cost[d] += row['think_tokens'] + row['answer_tokens']
        for n in ('requests.jsonl', 'responses.jsonl'):
            hashes[str((root / n).relative_to(history))] = sha(root / n)
        predictions = {d: vote(rs) for d, rs in found.items()}
    else:
        names = ['N1/gen.jsonl', 'P3/gen.jsonl'] if dataset == 'CTB' else ['P2/gen.jsonl']
        seen = set()
        for name in names:
            path = history / name; hashes[name] = sha(path)
            for row in read_rows(path):
                d = row['doc_id']
                if d not in docs:
                    continue
                key = (name, d, row['condition'], row['pass'])
                if key in seen:
                    raise ValueError('duplicated historical read')
                seen.add(key)
                ps = None
                if row.get('finish_reason') == 'stop' and '</think>' in row.get('text', ''):
                    try:
                        ps = {(a, b) for a, label, b in parse_triples(row['text'].rsplit('</think>', 1)[1]) if label != 'norel'}
                        if dataset == 'CTB':
                            ps = {tuple(sorted(p)) for p in ps}
                    except ValueError:
                        pass
                found[d].append(ps); invalid[d] += ps is None
                raw_cost[d] += row['num_tokens']
        expected = 10 if dataset == 'CTB' else 3
        if set(found) != set(docs) or any(len(rs) != expected for rs in found.values()):
            raise ValueError('incomplete historical benchmark')
        predictions = {}
        for d, rs in found.items():
            valid = [r for r in rs if r is not None]
            if dataset == 'CTB':
                predictions[d] = vote(valid) & pairs(docs[d])
            else:
                predictions[d] = {p for p in pairs(docs[d]) if valid and
                    sum((p in r) + (p[::-1] in r) for r in valid) / (2 * len(valid)) >= .5}
    support, denominators = {}, {}
    if dataset != 'MAVEN':
        for d, rs in found.items():
            valid = [r for r in rs if r is not None]
            denominators[d] = len(valid)
            support[d] = {','.join(p): sum((p in r) if dataset=='CTB' else
                       ((p in r)+(p[::-1] in r))/2 for r in valid) for p in sorted(pairs(docs[d]))}
    snapshot = dict(predictions={d: sorted(ps) for d, ps in predictions.items()},
                    support=support, valid_readings=denominators,
                    emitted_tokens=dict(raw_cost), invalid=dict(invalid), source_sha256=hashes,
                    scope='Historical cached numerical protocol; not a matched fresh engine baseline.')
    return snapshot


def prepare(args):
    out, source, history = Path(args.out), Path(args.source), Path(args.history)
    out.mkdir(parents=True, exist_ok=True)
    frozen = json.loads((HERE / 'wsp_confirmation_plan.json').read_text())
    for name in ('directions.npz', 'directions.json', 'manifest.json'):
        if sha(source / name) != frozen['source_files_sha256'][name]:
            raise ValueError(f'W1 frozen artifact drift: {name}')
    parent = json.loads((source / 'manifest.json').read_text())
    docs, prompt, subsets, input_hashes = load_inputs(args.dataset, history)
    if args.dataset == 'CTB' and set(docs) & set(parent['split']['fit']):
        raise ValueError('CTB evaluation overlaps direction extraction')
    jobs = make_reading_jobs(args.dataset, {d: public(row) for d, row in docs.items()}, prompt)
    prior = historical(args.dataset, docs, history)
    # Check archived benchmark before any new inference, without selecting policies.
    per = {d: counts(symmetric(set(map(tuple, ps))), gold_pairs(args.dataset, docs[d]))
           for d, ps in prior['predictions'].items()}
    replay = metrics(per)
    target = {'CTB': 61.19402985074627, 'ESC': 42.38178633975482, 'MAVEN': 29.473684210526315}[args.dataset]
    if abs(100 * replay['f1'] - target) > .0001:
        raise ValueError(f'historical replay differs: {100*replay["f1"]} versus {target}')
    rules = prompt['system'].split('# Output Format')[0].split('If there are no mentions')[0].strip()
    plan = dict(version=VERSION, dataset=args.dataset, documents=sorted(docs), subsets=subsets,
        arms=[arm('base'), arm('cc')], readings=READS, route_cap=ROUTE_CAP, block_size=BLOCK,
        verification_thinking=VERIFY_BUDGET, policies=list(POLICIES),
        vote_threshold=.5, invalid_reading='empty, fixed denominator',
        verification='two event orientations; A/B direction agreement after canonicalization; invalid preserves vote',
        primary_comparison='cc5__base_repair minus base5__base_repair; also cc5 minus base5',
        generation='two stage, thinking stop at </think>, then constrained JSON; differs from W2 decode',
        historical_augmentation='pooled original support plus five matched new reads; also equal-family union/intersection/majority',
        ranking='all policies reported; exploratory max F1 is not an independent confirmation',
        bootstrap=dict(seed=SEED, draws=10000, unit='topic' if args.dataset == 'ESC' else 'document'),
        source_manifest=parent['id'], directions_sha256=sha(source / 'directions.npz'),
        data_sha256=input_hashes, historical_sha256=prior['source_sha256'],
        reading_jobs_sha256=digest(jobs), documents_sha256=digest(docs), historical_snapshot_sha256=digest(prior),
        rules_sha256=digest(rules),
        implementation_sha256={n: sha(HERE / n) for n in ('wsp_combo.py', 'weight_space.py', 'offline.py', 'wsp_pilot.py')},
        engine=dict(model=args.model, max_model_len=args.max_model_len, batch=args.batch,
                    gpu_memory_utilization=args.gpu_memory_utilization, vllm='0.32.0'),
        scope='Exploratory reused development; MAVEN extension exposure unaudited. No claim of faithful reasoning.',
        max_active_hours=args.max_hours, historical_replay=replay)
    plan['id'] = digest(plan)
    freeze(out / 'manifest.json', plan)
    freeze(out / 'documents.json', docs)
    freeze(out / 'reading_jobs.json', jobs)
    freeze(out / 'historical.json', prior)
    # Only annotation rules; no evaluated relations or gold coreference in generation inputs.
    freeze(out / 'rules.json', rules)
    if not (out / 'directions.npz').exists():
        shutil.copyfile(source / 'directions.npz', out / 'directions.npz')
    if sha(out / 'directions.npz') != plan['directions_sha256']:
        raise ValueError('copied directions drift')
    print('WSP_COMBO_PLAN ' + json.dumps(plan), flush=True)
    return plan


def routed_pairs(doc, readings, systems):
    """Gold-free: prioritize disagreements, then supported pairs near 50% support."""
    universe = pairs(doc)
    union = set().union(*(r or set() for r in readings), *systems.values()) & universe
    count = Counter(p for r in readings for p in (r or set()))
    ranked = sorted(union, key=lambda p: (
        -int(len({p in ps for ps in systems.values()}) > 1),
        abs(count[p] / max(1, len(readings)) - .5), digest([VERSION, doc['id'], p])))
    return ranked[:ROUTE_CAP], len(ranked)


def verification_jobs(dataset, docs, rules, groups, systems):
    jobs, routing = [], {}
    for d, doc in sorted(docs.items()):
        selected, available = routed_pairs(doc, groups[d]['base'] + groups[d]['cc'],
                                          {name: pred[d] for name, pred in systems.items()})
        routing[d] = dict(selected=selected, available=available, unverified=max(0, available-len(selected)))
        for start in range(0, len(selected), BLOCK):
            block = selected[start:start+BLOCK]
            for reverse in (False, True):
                shown = [p[::-1] if reverse else p for p in block]
                lines = '\n'.join(f'{i}: A=<{a}>, B=<{b}>' for i, (a, b) in enumerate(shown))
                messages = [dict(role='system', content=rules +
                    '\nClassify only the listed pairs. A means event A causes event B; B means event B '
                    'causes event A; N means no annotated causal relation. Check evidence for each pair '
                    'and distinguish causation from temporal order, association and coreference. '
                    'Do not relax the annotation rules. Return one JSON object mapping every index to A, B or N.'),
                    dict(role='user', content=f'Document:\n{doc["text"]}\n\nPairs:\n{lines}')]
                jobs.append(job(dict(dataset=dataset, doc_id=d, kind='verification', reverse=reverse,
                                     pairs=block, block=start//BLOCK), messages, block_schema(len(block)),
                                VERIFY_BUDGET, 512))
    return jobs, routing


def read_cache(path, jobs, plan, partial=False):
    expected = {(a, j['key']) for a in ('base', 'cc') for j in jobs}
    rows = {}
    for row in read_rows(path) if Path(path).exists() else []:
        key = (row['arm'], row['key'])
        if key in rows or key not in expected or row.get('manifest_id') != plan['id']:
            raise ValueError('duplicate, foreign, or stale result')
        rows[key] = row
    if not partial and set(rows) != expected:
        raise ValueError(f'incomplete result cache {len(rows)}/{len(expected)}')
    return rows


def load_bundle(out):
    out=Path(out)
    plan=json.loads((out/'manifest.json').read_text())
    if digest({k:v for k,v in plan.items() if k!='id'})!=plan['id']:
        raise ValueError('manifest fingerprint mismatch')
    values=[]
    for name, key in [('documents.json','documents_sha256'),('historical.json','historical_snapshot_sha256'),
                      ('reading_jobs.json','reading_jobs_sha256')]:
        value=json.loads((out/name).read_text())
        if digest(value)!=plan[key]:raise ValueError(f'input fingerprint mismatch: {name}')
        values.append(value)
    if digest(json.loads((out/'rules.json').read_text()))!=plan['rules_sha256']:
        raise ValueError('annotation rules changed')
    return plan, *values


def reading_groups(jobs, rows, docs):
    groups = {d: {'base': [], 'cc': []} for d in docs}
    for name in ('base', 'cc'):
        for j in jobs:
            row = rows[(name, j['key'])]
            pred = parse_document(row['answer']) if row['valid'] else None
            groups[j['meta']['doc_id']][name].append(pred)
    if any(len(v[a]) != READS for v in groups.values() for a in ('base', 'cc')):
        raise ValueError('each document needs exactly five reads in each arm')
    return groups


def vote_systems(groups, prior):
    systems = {name: {} for name in ('base5', 'cc5', 'mixed10')}
    for d, rs in groups.items():
        systems['base5'][d] = vote(rs['base'])
        systems['cc5'][d] = vote(rs['cc'])
        systems['mixed10'][d] = vote(rs['base'] + rs['cc'])
    if set(prior['predictions']) == set(groups):
        systems['historical'] = {d: set(map(tuple, ps)) for d, ps in prior['predictions'].items()}
        for name in ('base5', 'cc5'):
            # Equal-family fusion avoids presenting a cached aggregate as ten fresh reads.
            systems['historical_' + name + '_union'] = {
                d: systems['historical'][d] | systems[name][d] for d in groups}
            systems['historical_' + name + '_intersection'] = {
                d: systems['historical'][d] & systems[name][d] for d in groups}
        if prior.get('support'):
            for label, arms in [('plus_base5', ['base']), ('plus_cc5', ['cc']), ('plus_mixed10', ['base', 'cc'])]:
                pooled = {}
                for d, rs in groups.items():
                    new = [r for a in arms for r in rs[a]]
                    supports = Counter({tuple(p.split(',', 1)):v for p,v in prior['support'][d].items()})
                    supports.update(p for r in new for p in (r or set()))
                    denominator=prior['valid_readings'][d]+len(new)
                    pooled[d]={p for p,v in supports.items() if v/denominator>=.5}
                systems['historical_'+label]=pooled
        systems['three_family_majority'] = {d: vote([systems[n][d] for n in ('historical', 'base5', 'cc5')],
                                                         threshold=2/3) for d in groups}
    return systems


def judge_decisions(jobs, rows):
    labels = defaultdict(dict)
    for a in ('base', 'cc'):
        for j in jobs:
            row, meta = rows[(a, j['key'])], j['meta']
            parsed = parse_block(row['answer'], len(meta['pairs'])) if row['valid'] else None
            for i, p in enumerate(meta['pairs']):
                label = parsed[str(i)] if parsed else None
                if meta['reverse'] and label in ('A', 'B'):
                    label = {'A': 'B', 'B': 'A'}[label]
                key = (meta['doc_id'], tuple(p))
                slot = (a, meta['reverse'])
                if slot in labels[key]:
                    raise ValueError('duplicate judge observation')
                labels[key][slot] = label
    result = {}
    for key, panel in labels.items():
        decision = {}
        for name, slots in [('base', [('base', False), ('base', True)]),
                            ('cc', [('cc', False), ('cc', True)]),
                            ('joint', [(a, r) for a in ('base', 'cc') for r in (False, True)])]:
            vals = [panel.get(s) for s in slots]
            decision[name] = vals[0] if vals[0] is not None and len(set(vals)) == 1 else None
        result[key] = decision
    return result


def combine(base, decisions, policy):
    result = {d: set(ps) for d, ps in base.items()}
    for (d, p), panel in decisions.items():
        if policy == 'strict_add_base_filter':
            if p in base[d] and panel['base'] == 'N':
                result[d].discard(p)
            elif p not in base[d] and panel['joint'] in ('A', 'B'):
                result[d].add(p)
            continue
        judge, action = policy.split('_')
        label = panel[judge]
        if label in ('A', 'B') and action != 'filter':
            result[d].add(p)
        elif label == 'N' and action != 'add':
            result[d].discard(p)
    return result


def paired(base, candidate, dataset):
    units = defaultdict(list)
    for d in sorted(base):
        units[d.split('_', 1)[0] if dataset == 'ESC' else d].append(d)
    arrays = [np.array([[sum(per[d][k] for d in ids) for k in ('tp', 'fp', 'fn')]
                       for ids in units.values()]) for per in (base, candidate)]
    ix = np.random.default_rng(SEED).integers(0, len(units), (10000, len(units)))
    vals = []
    for a in arrays:
        total = a[ix].sum(1); den = 2*total[:, 0]+total[:, 1]+total[:, 2]
        vals.append(np.divide(200*total[:, 0], den, out=np.zeros(10000), where=den > 0))
    return dict(delta_f1_pp=100*(metrics(candidate)['f1']-metrics(base)['f1']),
                ci95_pp=np.quantile(vals[1]-vals[0], [.025, .975]).tolist(),
                resampling_unit='topic' if dataset == 'ESC' else 'document', units=len(units))


def evaluate(dataset, docs, systems):
    result = {}
    for name, predictions in systems.items():
        per, outside = {}, 0
        for d, doc in docs.items():
            universe = pairs(doc); selected = predictions[d]
            outside += 2*len(selected-universe)
            per[d] = counts(symmetric(selected & universe), gold_pairs(dataset, doc))
        measured = metrics(per)
        penalty = dict(measured, fp=measured['fp']+outside)
        measured.update(documents=len(docs), per_document=per, outside_candidate_predictions=outside,
            f1_with_outside_penalty=2*penalty['tp']/max(1, 2*penalty['tp']+penalty['fp']+penalty['fn']))
        result[name] = measured
    reference = result['base5']['per_document']
    for value in result.values():
        value['vs_base5'] = paired(reference, value['per_document'], dataset)
    return result


def score(args, verification=False):
    out = Path(args.out); plan, docs, prior, jobs=load_bundle(out)
    rows = read_cache(out/'readings.jsonl', jobs, plan)
    groups = reading_groups(jobs, rows, docs)
    systems = vote_systems(groups, prior)
    if verification:
        judge_jobs = json.loads((out/'verification_jobs.json').read_text())
        expected_jobs, _=verification_jobs(args.dataset,{d:public(row) for d,row in docs.items()},
                                           json.loads((out/'rules.json').read_text()),groups,systems)
        if digest(judge_jobs)!=digest(expected_jobs):
            raise ValueError('verification prompts/routing changed')
        answers = read_cache(out/'verification.jsonl', judge_jobs, plan)
        decisions = judge_decisions(judge_jobs, answers)
        augmented = {f'{name}__{policy}': combine(pred, decisions, policy)
                     for name, pred in systems.items() for policy in POLICIES}
        systems.update(augmented)
    else:
        answers = {}
    results = evaluate(args.dataset, docs, systems)
    comparisons = {'cc5_vs_base5': paired(results['base5']['per_document'], results['cc5']['per_document'], args.dataset)}
    if verification:
        comparisons['cc5_base_repair_vs_base5_base_repair'] = paired(
            results['base5__base_repair']['per_document'], results['cc5__base_repair']['per_document'], args.dataset)
    if 'historical' in results:
        for value in results.values():
            value['vs_historical'] = paired(results['historical']['per_document'], value['per_document'], args.dataset)
    subsets = {}
    for label, ids in plan['subsets'].items():
        sub = {d: docs[d] for d in ids}
        sub_systems = {n: {d: p[d] for d in ids} for n, p in systems.items()}
        if set(ids) == set(prior['predictions']):
            sub_systems['historical'] = {d: set(map(tuple, prior['predictions'][d])) for d in ids}
        subsets[label] = evaluate(args.dataset, sub, sub_systems)
    costs = {stage: {a: dict(responses=sum(r['arm']==a for r in rs.values()),
             emitted_tokens=sum(r['num_tokens'] for r in rs.values() if r['arm']==a),
             invalid=sum(not r['valid'] for r in rs.values() if r['arm']==a)) for a in ('base', 'cc')}
             for stage, rs in [('readings', rows), ('verification', answers)]}
    for name,value in results.items():
        root, _, policy=name.partition('__')
        needs_historical='historical' in root or root=='three_family_majority'
        if policy:
            # Routing uses BOTH five-read families and historical predictions when
            # available, even for a nominal five-read verification output.
            new_tokens=sum(v['emitted_tokens'] for v in costs['readings'].values())
            needs_historical=set(prior['predictions'])==set(docs)
            judges=['base'] if policy.startswith('base_') else ['cc'] if policy=='cc_repair' else ['base','cc']
            new_tokens+=sum(costs['verification'][a]['emitted_tokens'] for a in judges)
        else:
            needed=['base'] if root=='base5' or 'base5' in root else ['cc'] if root=='cc5' or 'cc5' in root else []
            if root in ('mixed10','three_family_majority','historical_plus_mixed10'):needed=['base','cc']
            new_tokens=sum(costs['readings'][a]['emitted_tokens'] for a in needed)
        value['emitted_token_cost']=dict(new=new_tokens,
            historical=sum(prior['emitted_tokens'].values()) if needs_historical else 0,
            note='Counts all reads needed for shared routing; excludes prefill, extraction and model loading. Not wall time.')
    report = dict(manifest_id=plan['id'], dataset=args.dataset, complete=verification,
        status='complete' if verification else 'readings_complete', results=results, subsets=subsets,
        primary_comparisons=comparisons, costs=costs,
        historical_emitted_tokens=sum(prior['emitted_tokens'].values()),
        leaderboard=sorted(results, key=lambda n: (-results[n]['f1'], n)),
        caveat='All variants exploratory, correlated and selected on reused data; intervals unadjusted. '
               'No automatic promotion, no held-out or direction-specific claim.',
        run_id=os.environ.get('ORX_RUN_ID'), commit=os.environ.get('ORX_COMMIT'))
    write_json(out/('report.json' if verification else 'reading_report.json'), report)
    write_json(out/('predictions.json' if verification else 'reading_predictions.json'),
               {n: {d: sorted(ps) for d, ps in p.items()} for n, p in systems.items()})
    compact = dict(report, results={n: {k: v for k, v in r.items() if k != 'per_document'} for n, r in results.items()},
                   subsets={label: {n: {k:v for k,v in r.items() if k != 'per_document'} for n,r in values.items()}
                            for label, values in subsets.items()})
    print('WSP_COMBO_RESULT ' + json.dumps(compact), flush=True)
    return groups, systems, report


def preflight(args):
    from transformers import AutoTokenizer
    from offline import render
    plan = prepare(args)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    root = Path(args.model)
    identity = dict(config_sha256=sha(root/'config.json'),
        index_sha256=sha(root/'model.safetensors.index.json') if (root/'model.safetensors.index.json').exists() else None,
        weight_files={p.name:p.stat().st_size for p in sorted(root.glob('*.safetensors'))},
        limitation='Config/index hashes and weight sizes; not a full checkpoint content hash.')
    freeze(Path(args.out)/'model_identity.json', identity)
    jobs = json.loads((Path(args.out)/'reading_jobs.json').read_text())
    check_lengths(tok, jobs, args.max_model_len)
    print('WSP_COMBO_PREFLIGHT_OK ' + json.dumps(dict(manifest_id=plan['id'], documents=len(plan['documents']),
          reading_requests=2*len(jobs), max_verification_requests=4*len(plan['documents'])*math.ceil(ROUTE_CAP/BLOCK))), flush=True)


def check_lengths(tok, jobs, limit):
    from offline import render
    lengths = [len(tok.encode(render(tok, j['messages']), add_special_tokens=False)) for j in jobs]
    for n, j in zip(lengths, jobs):
        if n + j['thinking'] + j['answer_tokens'] + 256 >= limit:
            raise ValueError(f'context exceeds {limit}: {j["meta"]}, input={n}; no silent exclusion')
    print('WSP_COMBO_LENGTHS ' + json.dumps(dict(requests=len(jobs), max_prompt=max(lengths, default=0))), flush=True)


def run_batches(args, llm, tok, writers, vectors, jobs, path, plan, deadline):
    from offline import render
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    done = read_cache(path, jobs, plan, partial=True)
    for name in ('base', 'cc'):
        apply_arm(writers, vectors, arm(name))
        pending = [j for j in jobs if (name, j['key']) not in done]
        for start in range(0, len(pending), args.batch):
            if time.monotonic() > deadline:
                raise TimeoutError('active pilot budget reached; caches retained, no incomplete F1 promotion')
            batch = pending[start:start+args.batch]
            texts = [render(tok, j['messages']) for j in batch]
            parameters = [SamplingParams(temperature=0., top_p=1., max_tokens=j['thinking'],
                stop=['</think>'], include_stop_str_in_output=False) for j in batch]
            before = time.monotonic()
            traces = llm.generate(texts, parameters, use_tqdm=False)
            final_prompts = [s + r.outputs[0].text + '\n</think>\n\n' for s, r in zip(texts, traces, strict=True)]
            if any(len(tok.encode(p, add_special_tokens=False))+j['answer_tokens'] >= args.max_model_len
                   for p,j in zip(final_prompts,batch)):
                raise ValueError('answer context overflow')
            finals = llm.generate(final_prompts, [SamplingParams(temperature=0., top_p=1., max_tokens=j['answer_tokens'],
                    structured_outputs=StructuredOutputsParams(json=j['schema'])) for j in batch], use_tqdm=False)
            with Path(path).open('a') as stream:
                for j, trace, final in zip(batch, traces, finals, strict=True):
                    t, f = trace.outputs[0], final.outputs[0]
                    parsed = parse_document(f.text) if j['meta']['kind']=='reading' else parse_block(f.text, len(j['meta']['pairs']))
                    row = dict(key=j['key'], arm=name, manifest_id=plan['id'],
                        valid=f.finish_reason=='stop' and parsed is not None,
                        answer=f.text, thought=t.text, thinking_tokens=len(t.token_ids), answer_tokens=len(f.token_ids),
                        num_tokens=len(t.token_ids)+len(f.token_ids), thinking_forced=t.finish_reason=='length',
                        finish_reason=f.finish_reason, run_id=os.environ.get('ORX_RUN_ID'), commit=os.environ.get('ORX_COMMIT'))
                    stream.write(json.dumps(row, ensure_ascii=False)+'\n'); stream.flush()
            print('WSP_COMBO_PROGRESS ' + json.dumps(dict(stage=Path(path).stem, arm=name,
                completed=start+len(batch), pending_at_start=len(pending), batch_seconds=round(time.monotonic()-before, 2))), flush=True)


def generate(args):
    import vllm
    from vllm import SamplingParams
    from offline import build_llm, render
    if vllm.__version__ != '0.32.0':
        raise RuntimeError('verified vLLM 0.32.0 required')
    out = Path(args.out)
    plan, _, _, _ = load_bundle(out)
    if sha(out/'directions.npz') != plan['directions_sha256']:
        raise ValueError('direction fingerprint mismatch')
    # Charge active attempts across resumption, not time waiting for the shared GPU.
    previous = sum(json.loads(p.read_text())['seconds'] for p in out.glob('active-*.json'))
    started = time.monotonic(); deadline=started+max(0, args.max_hours*3600-previous)
    wall_path = out/f'active-{os.environ.get("ORX_RUN_ID", "local")}.json'
    writers = install_writers()
    try:
        llm, _, tok = build_llm(args.model, args.max_model_len, args.gpu_memory_utilization, max_num_seqs=args.batch)
        if set(writers) != set(LAYERS):
            raise RuntimeError('missing rank-one hooks')
        vectors = np.load(out/'directions.npz', allow_pickle=False)
        jobs = json.loads((out/'reading_jobs.json').read_text())
        check_lengths(tok, jobs, args.max_model_len)
        smoke_text = render(tok, jobs[0]['messages'])
        probes = []
        for a in ('base', 'cc', 'base'):
            apply_arm(writers, vectors, arm(a))
            probes.append(llm.generate([smoke_text], SamplingParams(temperature=0., max_tokens=2, logprobs=20), use_tqdm=False)[0].outputs[0])
        b,e,r = [p.logprobs[0] for p in probes]
        delta=max((abs(b[k].logprob-e[k].logprob) for k in b.keys() & e.keys()), default=0.)
        reset=max((abs(b[k].logprob-r[k].logprob) for k in b.keys() & r.keys()), default=float('inf'))
        if delta<=1e-5 or reset>1e-5 or probes[0].token_ids!=probes[2].token_ids:
            raise RuntimeError(f'edit smoke failed {delta=} {reset=}')
        write_json(out/'runtime_smoke.json', dict(edit_logprob_delta=delta, reset_delta=reset, sites=sorted(writers)))
        print('WSP_COMBO_SMOKE ' + json.dumps(dict(edit_logprob_delta=delta, reset_delta=reset)), flush=True)
        run_batches(args,llm,tok,writers,vectors,jobs,out/'readings.jsonl',plan,deadline)
        groups, systems, report = score(args)
        if os.environ.get('ORX_NOTIFY_FILE'):
            with open(os.environ['ORX_NOTIFY_FILE'],'a') as f:
                f.write(f"{args.dataset}: dix lectures terminées; vérification des désaccords en cours.\n")
        docs=json.loads((out/'documents.json').read_text()); rules=json.loads((out/'rules.json').read_text())
        judge_jobs, routes=verification_jobs(args.dataset,{d:public(row) for d,row in docs.items()},rules,groups,systems)
        freeze(out/'verification_jobs.json',judge_jobs);freeze(out/'routing.json',routes)
        check_lengths(tok,judge_jobs,args.max_model_len)
        print('WSP_COMBO_ROUTING '+json.dumps(dict(requests=2*len(judge_jobs),
              selected_pairs=sum(len(v['selected']) for v in routes.values()),
              unverified_pairs=sum(v['unverified'] for v in routes.values()))),flush=True)
        run_batches(args,llm,tok,writers,vectors,judge_jobs,out/'verification.jsonl',plan,deadline)
        _,_,report=score(args,verification=True)
        write_json(out/'terminal.json',dict(status='complete',report_sha256=sha(out/'report.json')))
        print('WSP_COMBO_COMPLETE '+json.dumps(dict(dataset=args.dataset,manifest_id=plan['id'],automatic_promotion=False)),flush=True)
    finally:
        write_json(wall_path,dict(seconds=time.monotonic()-started,run_id=os.environ.get('ORX_RUN_ID')))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=('prepare','preflight','generate','score'))
    p.add_argument('--dataset',choices=('CTB','ESC','MAVEN'),required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--source',default='/app/local_extensions/steering/mech/out/wsp-causal-v1-20261110')
    p.add_argument('--history',default='/app/local_extensions/steering/mech/out')
    p.add_argument('--model',default=MODEL)
    p.add_argument('--max-model-len',type=int,default=24576)
    p.add_argument('--batch',type=int,default=4)
    p.add_argument('--gpu-memory-utilization',type=float,default=.90)
    p.add_argument('--max-hours',type=float,default=18)
    args=p.parse_args()
    if args.stage=='score':score(args,verification=True)
    else:globals()[args.stage](args)


if __name__=='__main__':main()
