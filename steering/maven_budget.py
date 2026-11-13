"""Frozen MAVEN200 2x2: 20k/40k thinking x 2k/4k JSON, one GPU per arm.

prepare imports the historical manifest verbatim; run produces/reuses immutable
thinking traces; score refuses incomplete panels; compare performs paired analysis.
No evaluation labels enter requests. All persistent run data live on SPS.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import time

from steering import campaign_api as api
from steering import maven_pool as mp
from steering import validation_protocol as vp
from steering.compare import micro_f1

VERSION = 'maven-budget-20261113-v1'
ARM_SPEC = {'A': (20480, 2048, None), 'B': (20480, 4096, 'A'),
            'C': (40960, 2048, None), 'D': (40960, 4096, 'C')}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, value, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        if read(path) != value:
            raise ValueError(f'immutable file changed: {path}')
        return
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False, suffix='.tmp') as stream:
        tmp = Path(stream.name)
        stream.write(api.canonical(value) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def config(path):
    cfg = read(path)
    expected = ARM_SPEC[cfg['arm']]
    if cfg['protocol'] != VERSION or (cfg['thinking_tokens'], cfg['answer_tokens'], cfg['source_arm']) != expected:
        raise ValueError('arm does not match frozen factorial')
    if cfg['engine']['max_model_len'] != 65536 or cfg['documents'] != 200 or cfg['passes'] != 5:
        raise ValueError('panel/context differs from preregistration')
    return cfg


def output_dir(cfg):
    return Path(cfg['campaign_root']) / cfg['arm']


def freeze_engine(cfg):
    """Hash the actual model and SIF, before reserving a GPU; no mutable path-only identity."""
    model = Path(cfg['model_path'])
    paths = sorted(list(model.glob('*.safetensors')) + list(model.glob('*.json')))
    if not any(p.suffix == '.safetensors' for p in paths):
        raise ValueError('missing model weights')
    identity = {'container_sha256': sha(cfg['container']), 'files': {p.name: sha(p) for p in paths},
                'engine': cfg['engine'], 'served_model': cfg['model']}
    root = Path(cfg['campaign_root'])
    root.mkdir(parents=True, exist_ok=True)
    # Each arm verifies the actual bytes; shared identity may never silently change.
    with (root / 'engine.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        atomic(root / 'engine_identity.json', identity, immutable=True)
    print('MB_ENGINE', api.canonical({'sha256': api.digest(identity), 'files': len(paths)}), flush=True)
    return identity


def import_jobs(cfg, historical, docs):
    selected = historical['selected']
    if len(selected) != cfg['documents'] or len(set(selected)) != len(selected) or set(selected) != set(docs):
        raise ValueError('document membership differs from frozen manifest')
    if len(historical['demo_ids']) != cfg['passes']:
        raise ValueError('demo bank changed')
    source_jobs = mp.read_jsonl(Path(cfg['historical_root']) / 'jobs.jsonl')
    if api.digest([j['key'] for j in source_jobs]) != historical['jobs_sha256']:
        raise ValueError('historical job hash mismatch')
    jobs = []
    for old in source_jobs:
        if old['budget'] != 'long':
            continue
        if old['doc_id'] not in docs or old['demo_id'] != historical['demo_ids'][old['pass_id']]:
            raise ValueError('document/demo association changed')
        job = {k: old[k] for k in ('doc_id', 'pass_id', 'demo_id', 'messages')}
        job['key'] = api.digest(job)
        jobs.append(job)
    if len(jobs) != cfg['documents'] * cfg['passes'] or len({j['key'] for j in jobs}) != len(jobs):
        raise ValueError('missing/duplicate document-demo job')
    if {(j['doc_id'], j['pass_id']) for j in jobs} != {(d, i) for d in selected for i in range(cfg['passes'])}:
        raise ValueError('incomplete five-demo panel')
    return sorted(jobs, key=lambda j: (j['doc_id'], j['pass_id']))


def prepare(cfg):
    out, history = output_dir(cfg), Path(cfg['historical_root'])
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'complete.json').exists():
        raise ValueError('arm already complete; do not allocate another GPU')
    for name, expected in cfg['historical_sha256'].items():
        if sha(history / name) != expected:
            raise ValueError(f'historical input changed: {name}')
    old, docs = read(history / 'manifest.json'), read(history / 'documents.json')
    if old['revision'] != mp.MAVEN_REVISION or old['prompt_sha256'] != sha(mp.PROMPT):
        raise ValueError('dataset revision or prompt changed')
    jobs = import_jobs(cfg, old, docs)
    engine = freeze_engine(cfg)
    manifest = {'protocol': VERSION, 'configuration': cfg, 'engine_sha256': api.digest(engine),
                'historical_sha256': cfg['historical_sha256'], 'selected': old['selected'],
                'demo_ids': old['demo_ids'], 'jobs_sha256': api.digest(jobs), 'expected_reads': len(jobs),
                'dataset_revision': old['revision'], 'prompt_sha256': old['prompt_sha256']}
    for name, value in [('manifest.json', manifest), ('jobs.json', jobs), ('documents.json', docs),
                        ('doc_structure.json', read(history / 'doc_structure.json')),
                        ('engine_identity.json', engine)]:
        atomic(out / name, value, immutable=True)
    if cfg['source_arm']:
        source_bundle(cfg, manifest, jobs)  # fail before acquiring a GPU
    print('MB_PLAN', api.canonical({'arm': cfg['arm'], 'jobs': len(jobs),
          'manifest_sha256': sha(out / 'manifest.json'), 'configuration': cfg}), flush=True)
    return manifest, jobs


def source_bundle(cfg, manifest, jobs):
    source = Path(cfg['campaign_root']) / cfg['source_arm']
    expected = cfg.get('source_completion_sha256')
    if not expected or sha(source / 'complete.json') != expected:
        raise ValueError('source completion must be pinned in committed configuration')
    complete, original = read(source / 'complete.json'), read(source / 'manifest.json')
    if original['configuration']['experiment_id'] != cfg['source_experiment_id']:
        raise ValueError('wrong source experiment')
    if complete['manifest_sha256'] != sha(source / 'manifest.json') or complete['reads'] != len(jobs):
        raise ValueError('incomplete or changed source manifest')
    for field in ('jobs_sha256', 'engine_sha256', 'historical_sha256'):
        if original[field] != manifest[field]:
            raise ValueError(f'source incompatibility: {field}')
    if original['configuration']['thinking_tokens'] != cfg['thinking_tokens']:
        raise ValueError('source thinking budget mismatch')
    expected_keys = {j['key'] for j in jobs}
    if set(complete['traces']) != expected_keys or set(complete['answers']) != expected_keys:
        raise ValueError('incomplete source panel')
    for kind in ('traces', 'answers'):
        for key, expected_hash in complete[kind].items():
            if sha(source / kind / (key + '.json')) != expected_hash:
                raise ValueError(f'source {kind} changed: {key}')
    return source, complete


def trace_contract(cfg, manifest, job):
    return api.digest({'protocol': VERSION, 'engine': manifest['engine_sha256'],
                       'jobs': manifest['jobs_sha256'], 'job': job['key'],
                       'thinking_tokens': cfg['thinking_tokens']})


def validate_trace(record, contract):
    if record['contract'] != contract or record['sha256'] != api.digest(record['data']):
        raise ValueError('trace identity/content mismatch')
    return record['data']


def answer_contract(cfg, trace):
    return api.digest({'trace_sha256': trace['sha256'], 'trace_contract': trace['contract'],
                       'answer_tokens': cfg['answer_tokens'], 'schema': vp.document_schema()})


class Reader:
    def __init__(self, cfg, manifest, jobs, url):
        from transformers import AutoTokenizer
        self.cfg, self.manifest, self.out = cfg, manifest, output_dir(cfg)
        self.source = source_bundle(cfg, manifest, jobs)[0] if cfg['source_arm'] else self.out
        self.tok = AutoTokenizer.from_pretrained(cfg['model_path'], trust_remote_code=True)
        self.client = api.Client(url, os.environ.get('MB_API_KEY', ''), self.out / 'http-cache',
                                 api.digest(manifest), timeout=21600)
        models, _, _ = self.client.call('/models')
        entry = next(m for m in models['data'] if m['id'] == cfg['model'])
        if entry.get('max_model_len') != cfg['engine']['max_model_len']:
            raise ValueError(f'server context mismatch: {entry.get("max_model_len")}')
        for job in jobs:
            _, before = mp.prompt_tokens(self.tok, job['messages'])
            # Reserve the largest final budget even for 2k arms, so both can reuse the trace.
            if len(before) + cfg['thinking_tokens'] + 4096 + 32 > cfg['engine']['max_model_len']:
                raise ValueError('frozen prompt does not fit; refusing truncation or exclusion')
        probe, ids = mp.prompt_tokens(self.tok, [{'role': 'user', 'content': 'Return JSON.'}])
        obj, _, _ = self.client.call('/../tokenize', {'model': cfg['model'], 'prompt': probe,
                                                    'add_special_tokens': False})
        if obj['tokens'] != ids:
            raise ValueError('server/client tokenizer mismatch')

    def smoke(self):
        """Check full-context prefill, schema and 32-way scheduling without evaluation labels."""
        messages = [{'role': 'user', 'content': 'Return exactly {"relations": []}.'}]
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                            enable_thinking=False)
        short = self.tok.encode(text, add_special_tokens=False)
        filler = self.tok.encode('This is neutral padding. ', add_special_tokens=False)
        long = (filler * (45000 // len(filler) + 1))[:45000] + short

        def probe(ids):
            body = api.payload(self.cfg['model'], [ids], 64, 'direct')
            body['structured_outputs']['json'] = vp.document_schema()
            result, _, _ = self.client.call('/completions', body)
            final = api.ordered_choices(result, 1)[0]
            if final['finish_reason'] != 'stop' or vp.document_parse(final['text']) is None:
                raise ValueError('synthetic structured-output smoke failed')

        probe(long)
        with ThreadPoolExecutor(max_workers=self.cfg['engine']['concurrency']) as pool:
            list(pool.map(probe, [short] * self.cfg['engine']['concurrency']))
        print('MB_SMOKE', api.canonical({'context_tokens': len(long),
              'concurrent_requests': self.cfg['engine']['concurrency'], 'status': 'passed'}), flush=True)

    def one(self, job):
        cfg, key = self.cfg, job['key']
        contract = trace_contract(cfg, self.manifest, job)
        trace_path = self.source / 'traces' / (key + '.json')
        tick = time.monotonic()
        trace_hit = trace_path.exists()
        if trace_hit:
            trace_record = read(trace_path)
            trace = validate_trace(trace_record, contract)
        else:
            if cfg['source_arm']:
                raise ValueError('source trace missing; regeneration is forbidden')
            text, before = mp.prompt_tokens(self.tok, job['messages'])
            result, cache_hit, elapsed = self.client.call('/completions',
                api.payload(cfg['model'], [before], cfg['thinking_tokens']), cache=True)
            choice = api.ordered_choices(result, 1)[0]
            if choice['finish_reason'] not in ('stop', 'length'):
                raise ValueError(f'unexpected thinking finish: {choice["finish_reason"]}')
            # Keep the historical text->token protocol exactly; B/D consume these exact IDs.
            after = self.tok.encode(text + choice['text'] + '\n</think>\n\n', add_special_tokens=False)
            if len(after) + 4096 > cfg['engine']['max_model_len']:
                raise ValueError('final prompt exceeds context')
            trace = {'key': key, 'doc_id': job['doc_id'], 'pass_id': job['pass_id'],
                     'text': choice['text'], 'token_ids': choice.get('token_ids'),
                     'final_prompt_ids': after, 'prompt_tokens': len(before),
                     'think_tokens': len(choice.get('token_ids') or self.tok.encode(choice['text'], add_special_tokens=False)),
                     'thinking_forced': choice.get('stop_reason') != '</think>',
                     'finish_reason': choice['finish_reason'], 'stop_reason': choice.get('stop_reason'),
                     'generation_seconds': elapsed, 'api_usage': result.get('usage', {}),
                     'source_run': os.environ['ORX_RUN_ID'], 'source_commit': os.environ['ORX_COMMIT']}
            trace_record = {'contract': contract, 'sha256': api.digest(trace), 'data': trace}
            atomic(trace_path, trace_record, immutable=True)
        target = self.out / 'answers' / (key + '.json')
        identity = answer_contract(cfg, trace_record)
        if target.exists():
            row = read(target)
            if row['contract'] != identity:
                raise ValueError('answer cache identity mismatch')
            return row, True
        body = api.payload(cfg['model'], [trace['final_prompt_ids']], cfg['answer_tokens'], 'direct')
        body['structured_outputs']['json'] = vp.document_schema()
        result, cache_hit, elapsed = self.client.call('/completions', body, cache=True)
        final = api.ordered_choices(result, 1)[0]
        obj = vp.document_parse(final['text']) if final['finish_reason'] == 'stop' else None
        row = {'key': key, 'contract': identity, 'doc_id': job['doc_id'], 'pass_id': job['pass_id'],
               'arm': cfg['arm'], 'status': 'valid' if obj is not None else 'invalid_answer',
               'answer': final['text'], 'answer_finish': final['finish_reason'],
               'trace_sha256': trace_record['sha256'], 'thinking_forced': trace['thinking_forced'],
               'think_tokens': trace['think_tokens'], 'prompt_tokens': trace['prompt_tokens'],
               'final_prompt_tokens': len(trace['final_prompt_ids']),
               'answer_tokens': len(final.get('token_ids') or self.tok.encode(final['text'], add_special_tokens=False)),
               'generation_seconds': elapsed, 'elapsed_seconds': time.monotonic() - tick,
               'trace_reused': bool(cfg['source_arm']), 'trace_resume_hit': trace_hit,
               'answer_http_cache_hit': cache_hit, 'source_run': trace['source_run'],
               'answer_run': os.environ['ORX_RUN_ID'], 'answer_commit': os.environ['ORX_COMMIT']}
        atomic(target, row, immutable=True)
        return row, False


def run(cfg, url):
    out = output_dir(cfg)
    manifest, jobs = read(out / 'manifest.json'), read(out / 'jobs.json')
    if manifest['configuration'] != cfg:
        raise ValueError('configuration changed after prepare')
    if (out / 'complete.json').exists():
        raise ValueError('arm already complete')
    with (out / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        reader = Reader(cfg, manifest, jobs, url)
        reader.smoke()
        run_id = os.environ['ORX_RUN_ID']
        atomic(out / 'attempts' / (run_id + '.json'),
               {'run': run_id, 'commit': os.environ['ORX_COMMIT'], 'started_unix': time.time()})
        todo = iter(jobs)
        stats, errors, start = Counter(), [], time.monotonic()
        with ThreadPoolExecutor(max_workers=cfg['engine']['concurrency']) as pool:
            pending = {pool.submit(reader.one, job): job for job in
                       [next(todo) for _ in range(min(len(jobs), cfg['engine']['concurrency']))]}
            while pending:
                finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    job = pending.pop(future)
                    try:
                        row, hit = future.result()
                        stats['done'] += 1
                        stats['answer_cache_hits'] += int(hit)
                        stats['invalid'] += row['status'] != 'valid'
                        stats['thinking_forced'] += row['thinking_forced']
                        stats['answer_tokens'] += row['answer_tokens']
                        stats['think_tokens'] += row['think_tokens']
                    except Exception as exc:
                        errors.append({'key': job['key'], 'error': repr(exc)[:1200]})
                        print('MB_ERROR', api.canonical(errors[-1]), flush=True)
                if not errors:
                    for _ in range(len(finished)):
                        job = next(todo, None)
                        if job is not None:
                            pending[pool.submit(reader.one, job)] = job
                print('MB_PROGRESS', api.canonical({'arm': cfg['arm'], **stats,
                      'expected': len(jobs), 'elapsed_minutes': round((time.monotonic()-start)/60, 2),
                      'errors': len(errors)}), flush=True)
        attempt = read(out / 'attempts' / (run_id + '.json'))
        attempt.update(finished_unix=time.time(), inference_seconds=time.monotonic()-start, errors=errors,
                       status='failed' if errors else 'complete')
        atomic(out / 'attempts' / (run_id + '.json'), attempt)
        if errors:
            raise RuntimeError(f'{len(errors)} transport/contract errors; resume only missing work')
        return score(cfg)


def pairs_and_counts(jobs, rows, docs, structure, threshold):
    by_doc = defaultdict(list)
    quality = Counter()
    for job in jobs:
        universe = set(mp.c.pairs(docs[job['doc_id']]))
        by_doc[job['doc_id']].append(mp.reading_pairs(rows.get(job['key']), universe, quality))
    predictions = {d: mp.vote(by_doc[d], threshold) for d in docs}
    scopes = {name: {} for name in ('all', 'intra', 'inter', 'distance_1', 'distance_2_3', 'distance_ge4')}
    for doc, pred in predictions.items():
        gold = {tuple(sorted(p.split(','))) for p in docs[doc]['gold']}
        sentence = structure[doc]['sentence_of']
        for scope in scopes:
            def accepts(pair):
                dist = abs(sentence[pair[0]] - sentence[pair[1]])
                return (scope == 'all' or scope == 'intra' and dist == 0 or scope == 'inter' and dist > 0
                        or scope == 'distance_1' and dist == 1 or scope == 'distance_2_3' and 2 <= dist <= 3
                        or scope == 'distance_ge4' and dist >= 4)
            p, g = {x for x in pred if accepts(x)}, {x for x in gold if accepts(x)}
            scopes[scope][doc] = {'tp': len(p & g), 'fp': len(p-g), 'fn': len(g-p)}
    return predictions, scopes, dict(quality)


def metric(counts):
    total = {k: sum(row[k] for row in counts.values()) for k in ('tp', 'fp', 'fn')}
    tp, fp, fn = (total[k] for k in ('tp', 'fp', 'fn'))
    return {**total, 'precision': 100*tp/(tp+fp) if tp+fp else 0,
            'recall': 100*tp/(tp+fn) if tp+fn else 0,
            'f1': micro_f1(list(counts.values())), 'documents': len(counts)}


def score(cfg):
    out = output_dir(cfg)
    manifest, jobs = read(out / 'manifest.json'), read(out / 'jobs.json')
    docs, structure = read(out / 'documents.json'), read(out / 'doc_structure.json')
    if manifest['configuration'] != cfg or api.digest(jobs) != manifest['jobs_sha256']:
        raise ValueError('changed manifest/config/jobs')
    source = source_bundle(cfg, manifest, jobs)[0] if cfg['source_arm'] else out
    rows, trace_hashes, answer_hashes = {}, {}, {}
    for job in jobs:
        key = job['key']
        trace_file, answer_file = source / 'traces' / (key+'.json'), out / 'answers' / (key+'.json')
        record = read(trace_file)
        validate_trace(record, trace_contract(cfg, manifest, job))
        row = read(answer_file)  # missing reads prohibit final scoring
        if row['contract'] != answer_contract(cfg, record):
            raise ValueError('answer/trace mismatch at score')
        rows[key], trace_hashes[key], answer_hashes[key] = row, sha(trace_file), sha(answer_file)
    systems, per_doc, predictions = {}, {}, {}
    for threshold in (1, 2, 3, 4, 5):
        pred, counts, quality = pairs_and_counts(jobs, rows, docs, structure, threshold)
        name = f'vote_ge{threshold}'
        systems[name] = {scope: metric(values) for scope, values in counts.items()}
        per_doc[name] = counts
        if threshold == 3:
            predictions = {d: [list(p) for p in sorted(ps)] for d, ps in pred.items()}
    single = {}
    for pass_id in range(5):
        subset = [j for j in jobs if j['pass_id'] == pass_id]
        _, counts, _ = pairs_and_counts(subset, rows, docs, structure, 1)
        single[str(pass_id)] = metric(counts['all'])
    history = read(Path(cfg['historical_root']) / 'reads_compact.json')
    old_union = defaultdict(set)
    for row in history['reads']:
        if row['status'] == 'valid':
            old_union[row['doc']].update(tuple(sorted(p.split(','))) for p in row['pairs'])
    recovered = 0
    for doc, values in predictions.items():
        gold = {tuple(sorted(p.split(','))) for p in docs[doc]['gold']}
        recovered += len({tuple(p) for p in values} & (gold-old_union[doc]))
    attempts = [read(p) for p in (out / 'attempts').glob('*.json')]
    seconds = sum(a.get('inference_seconds', 0) for a in attempts)
    report = {'protocol': VERSION, 'arm': cfg['arm'], 'experiment_id': cfg['experiment_id'],
              'configuration': cfg, 'manifest_sha256': sha(out / 'manifest.json'),
              'reads_expected': len(jobs), 'reads_done': len(rows), 'primary': 'vote_ge3',
              'metrics': systems, 'single_pass_metrics': single, 'per_document': per_doc,
              'quality': quality, 'historical_never_found_recovered_at_vote3': recovered,
              'forced_closes': sum(x['thinking_forced'] for x in rows.values()),
              'answer_truncations': sum(x['answer_finish']=='length' for x in rows.values()),
              'cost': {'thinking_tokens_full_recipe': sum(x['think_tokens'] for x in rows.values()),
                       'thinking_tokens_incremental': 0 if cfg['source_arm'] else sum(x['think_tokens'] for x in rows.values()),
                       'answer_tokens': sum(x['answer_tokens'] for x in rows.values()),
                       'final_input_tokens': sum(x['final_prompt_tokens'] for x in rows.values()),
                       'gpu_inference_hours_all_attempts': seconds/3600,
                       'note': 'Allocation/setup hours recorded in Slurm accounting; B/D additionally require source costs.'},
              'runs': sorted({x['answer_run'] for x in rows.values()}),
              'scope': 'Paired comparison on previously exposed 200 documents; no held-out or directed claim.'}
    atomic(out / 'predictions.json', predictions)
    atomic(out / 'report.json', report)
    completion = {'protocol': VERSION, 'arm': cfg['arm'], 'experiment_id': cfg['experiment_id'],
                  'manifest_sha256': sha(out / 'manifest.json'), 'reads': len(rows),
                  'report_sha256': sha(out / 'report.json'), 'traces': trace_hashes,
                  'answers': answer_hashes, 'source_arm': cfg['source_arm'], 'runs': report['runs']}
    atomic(out / 'complete.json', completion, immutable=True)
    print('MB_METRIC', api.canonical({'arm': cfg['arm'], **systems['vote_ge3']['all']}), flush=True)
    print('MB_RESULT', api.canonical({k:v for k,v in report.items() if k!='per_document'}), flush=True)
    print('MB_COMPLETE', api.canonical({'path': str(out / 'complete.json'),
          'sha256': sha(out / 'complete.json')}), flush=True)
    return report


def paired_contrasts(reports, draws=5000, seed=20261112):
    names = ('D-B', 'C-A', 'B-A', 'D-C', '(D-C)-(B-A)')
    per = {arm: reports[arm]['per_document']['vote_ge3']['all'] for arm in 'ABCD'}
    docs = sorted(per['A'])
    if any(set(per[arm]) != set(docs) for arm in 'BCD'):
        raise ValueError('unpaired document sets')
    def values(ids):
        f = {arm: micro_f1([per[arm][d] for d in ids]) for arm in 'ABCD'}
        return (f['D']-f['B'], f['C']-f['A'], f['B']-f['A'], f['D']-f['C'],
                (f['D']-f['C'])-(f['B']-f['A']))
    delta, rng, samples = values(docs), random.Random(seed), [[] for _ in names]
    for _ in range(draws):
        for sample, value in zip(samples, values(rng.choices(docs, k=len(docs)))):
            sample.append(value)
    result = {}
    for name, point, sample in zip(names, delta, samples):
        sample.sort()
        result[name] = {'delta_f1': point, 'ci95': [sample[int(.025*draws)], sample[int(.975*draws)-1]],
                        'primary': name=='D-B', 'unit': 'document'}
    return result


def compare(cfg):
    root = Path(cfg['campaign_root'])
    reports = {}
    for arm in 'ABCD':
        complete = read(root / arm / 'complete.json')
        if sha(root / arm / 'report.json') != complete['report_sha256']:
            raise ValueError('report changed after completion')
        reports[arm] = read(root / arm / 'report.json')
    for producer, consumer in [('A','B'), ('C','D')]:
        c = reports[consumer]['configuration']
        if c['source_completion_sha256'] != sha(root / producer / 'complete.json'):
            raise ValueError('factorial traces are not paired')
    manifests = [read(root / arm / 'manifest.json') for arm in 'ABCD']
    for field in ('jobs_sha256', 'engine_sha256', 'historical_sha256'):
        if any(m[field] != manifests[0][field] for m in manifests):
            raise ValueError('incompatible factorial inputs')
    result = {'protocol': VERSION, 'primary': 'D-B', 'draws': cfg['bootstrap_draws'], 'seed': cfg['seed'],
              'contrasts': paired_contrasts(reports, cfg['bootstrap_draws'], cfg['seed']),
              'report_sha256': {a:sha(root/a/'report.json') for a in 'ABCD'},
              'metrics': {a:reports[a]['metrics'] for a in 'ABCD'},
              'secondary_intervals': 'Unadjusted; only D-B at vote_ge3 is the primary hypothesis.'}
    atomic(root / 'factorial_report.json', result)
    print('MB_FACTORIAL', api.canonical(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cmd', choices=('prepare', 'run', 'score', 'compare'))
    parser.add_argument('--config', default='steering/maven_budget_config.json')
    parser.add_argument('--url', default='http://127.0.0.1:8100/v1')
    args = parser.parse_args()
    cfg = config(args.config)
    if args.cmd == 'prepare':
        prepare(cfg)
    elif args.cmd == 'run':
        run(cfg, args.url)
    elif args.cmd == 'score':
        score(cfg)
    else:
        compare(cfg)


if __name__ == '__main__':
    main()
