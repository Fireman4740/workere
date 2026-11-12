"""N1+P3 on MAVEN200: five demo-diverse greedy document reads at two thinking budgets, pooled vote.

N1 = short reads (8192 thinking tokens), P3 = long reads (20480 thinking tokens). Both reuse the
VC3 MAVEN12 protocol exactly (same five train demos, instruction, two-stage /v1/completions:
greedy thinking stopped at </think>, then a structured JSON relations answer of 2048 tokens).

Subcommands (all resumable):
  prepare --out DIR                        CPU: download MAVEN, select 200 docs, freeze jobs.jsonl
  run --out DIR --url URL --shard i --shards n --concurrency k
                                           one vLLM server: run this shard's reads, one request
                                           per sequence so continuous batching keeps the GPU full
  score --out DIR                          votes, metrics, paired bootstrap -> report.json
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import random
import sys
import threading
import time

from steering import campaign_api as api
from steering import campaign_blocks as c
from steering import campaign_facts as f
from steering import campaign_adjudicate as a
from steering import validation_protocol as v
from steering.compare import paired_delta

VERSION = 'maven-pool-20261112-v1'
MAVEN_REPO = 'Nofing/MAVEN-ERE-standard-eci-workere'
MAVEN_REVISION = '830a58ee7a092d5ca27b4a8c28c18e2c82abf814'
BUDGETS = {'short': 8192, 'long': 20480}
ANSWER_TOKENS = 2048
MAX_MODEL_LEN = 32768
N_DOCS, QUARTILES = 200, 4
SEED = 20261112
PROMPT = Path('prompts/maven_ere_standard_eci_lean_coref.yaml')
MODEL = os.environ.get('API_MODEL', 'qwen3.8-27b')
TOKENIZER = os.environ.get('TOKENIZER_PATH', c.MODEL)


def load_split(split):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as parquet
    from eci.data import doc_from_row
    path = Path(hf_hub_download(MAVEN_REPO, f'data/{split}-00000-of-00001.parquet',
                                repo_type='dataset', revision=MAVEN_REVISION))
    rows = parquet.read_table(path).to_pylist()
    docs = [doc_from_row(row, i, text_field='text', ann_field='annots',
                         valid_labels={'causes', 'causedby', 'norel'}, binary_undirected=False,
                         shuffle_pair_list=False) for i, row in enumerate(rows)]
    return docs, dict(path=str(path), sha256=c.sha(path))


def demo_bank(train, test):
    """Identical to VC3 prepare_maven: five train demos disjoint by id and text from test."""
    test_ids = {doc.id for doc in test}
    text_hashes = {api.digest(doc.text) for doc in test}
    bank = sorted([doc for doc in train if doc.id not in test_ids and api.digest(doc.text) not in text_hashes
                   and len(doc.tokens) <= 400 and any(label != 'norel' for _, label, _ in doc.gold)],
                  key=lambda doc: api.digest([v.VERSION, 'demo', doc.id]))[:5]
    assert len(bank) == 5
    return bank


def instruction():
    import yaml
    system = yaml.safe_load(PROMPT.read_text())['system'].split('If there are no mentions')[0].strip()
    return (system + '\n\nReturn only a JSON object with a relations array of objects containing '
            'pair (two mention IDs separated by comma) and label (causes or causedby). '
            'Omit norel. Emit both orientations. Do not output prose or extra keys.')


def messages_for(doc, demo, system):
    answer = {'relations': [dict(pair=f'{x},{y}', label=lab) for x, lab, y in demo.gold if lab != 'norel']}
    return [dict(role='system', content=system), dict(role='user', content='Document:\n' + demo.text),
            dict(role='assistant', content=api.canonical(answer)), dict(role='user', content='Document:\n' + doc.text)]


def prompt_tokens(tok, messages):
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=True, preserve_thinking=True)
    return text, tok.encode(text, add_special_tokens=False)


def select_docs(test, lengths):
    """Gold-blind: 50 per prompt-length quartile among documents that fit the long budget."""
    fits = sorted([d for d in test if lengths[d.id] is not None], key=lambda d: (lengths[d.id], d.id))
    rng = random.Random(SEED)
    per = N_DOCS // QUARTILES
    chosen = []
    for k in range(QUARTILES):
        stratum = fits[k * len(fits) // QUARTILES:(k + 1) * len(fits) // QUARTILES]
        chosen.extend(rng.sample(stratum, per))
    return chosen


def prepare(out: Path):
    from transformers import AutoTokenizer
    out.mkdir(parents=True, exist_ok=True)
    (train, train_src), (test, test_src) = load_split('train'), load_split('test')
    assert len(test) == 710, 'benchmark revision split size changed'
    bank, system = demo_bank(train, test), instruction()
    tok = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)
    limit = MAX_MODEL_LEN - BUDGETS['long'] - ANSWER_TOKENS - 32
    lengths = {}
    for doc in test:  # longest demo bounds the prompt; exclusion is reported, never silent
        worst = max(len(prompt_tokens(tok, messages_for(doc, demo, system))[1]) for demo in bank)
        lengths[doc.id] = worst if worst <= limit else None
    chosen = select_docs(test, lengths)
    documents, jobs = {}, []
    for doc in chosen:
        positive = {(x, y) for x, lab, y in doc.gold if lab != 'norel'}
        positive |= {(y, x) for x, y in positive}
        assert positive <= set(doc.pair_list)
        documents[doc.id] = dict(id=doc.id, pair_list=doc.pair_list, gold=sorted(f'{x},{y}' for x, y in positive),
                                 prompt_tokens_max=lengths[doc.id])
        for budget, think in BUDGETS.items():
            for pass_id, demo in enumerate(bank):
                job = dict(protocol=VERSION, doc_id=doc.id, pass_id=pass_id, demo_id=demo.id, budget=budget,
                           thinking=think, answer_tokens=ANSWER_TOKENS, messages=messages_for(doc, demo, system))
                job['key'] = api.digest(job)
                jobs.append(job)
    excluded = sorted(d for d, n in lengths.items() if n is None)
    pilot12 = set(json.loads(os.environ.get('PILOT12_IDS', '[]')))
    manifest = dict(protocol=VERSION, dataset_repo=MAVEN_REPO, revision=MAVEN_REVISION,
                    sources=dict(train=train_src, test=test_src), prompt_sha256=c.sha(PROMPT),
                    demo_ids=[d.id for d in bank], budgets=BUDGETS, answer_tokens=ANSWER_TOKENS,
                    max_model_len=MAX_MODEL_LEN, selected=[d.id for d in chosen], seed=SEED,
                    excluded_too_long=excluded, overlap_with_pilot12=sorted(pilot12 & set(documents)),
                    jobs=len(jobs), jobs_sha256=api.digest([j['key'] for j in jobs]), tokenizer=TOKENIZER,
                    selection='50 per prompt-length quartile among docs fitting 32k with 20k thinking; seeded; gold-blind')
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    (out / 'documents.json').write_text(api.canonical(documents))
    with (out / 'jobs.jsonl').open('w') as sink:
        for job in jobs:
            sink.write(api.canonical(job) + '\n')
    print('MP_PLAN', api.canonical({k: manifest[k] for k in ('jobs', 'jobs_sha256', 'demo_ids')}),
          'excluded', len(excluded), flush=True)


def read_jsonl(path: Path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:  # a killed writer may leave one partial last line
            continue
    return rows


class Reader:
    """Runs reads against one server; the per-request disk cache makes every stage resumable."""

    def __init__(self, url: str, cache: Path):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)
        self.client = api.Client(url, os.environ.get('API_KEY', ''), cache,
                                 api.digest(dict(protocol=VERSION, model=MODEL)), timeout=7200)
        models, _, _ = self.client.call('/models')
        assert MODEL in {m['id'] for m in models['data']}, f'server does not serve {MODEL}'
        probe = self.tok.apply_chat_template([{'role': 'user', 'content': 'Return JSON.'}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=True, preserve_thinking=True)
        obj, _, _ = self.client.call('/../tokenize', dict(model=MODEL, prompt=probe, add_special_tokens=False))
        assert obj['tokens'] == self.tok.encode(probe, add_special_tokens=False), 'server tokenizer differs'

    def read(self, job):
        text, before = prompt_tokens(self.tok, job['messages'])
        assert len(before) + job['thinking'] + job['answer_tokens'] + 32 <= MAX_MODEL_LEN
        t0 = time.monotonic()
        first, hit1, _ = self.client.call('/completions', api.payload(MODEL, [before], job['thinking']), cache=True)
        trace = api.ordered_choices(first, 1)[0]
        after = self.tok.encode(text + trace['text'] + '\n</think>\n\n', add_special_tokens=False)
        body = api.payload(MODEL, [after], job['answer_tokens'], 'direct')
        body['structured_outputs']['json'] = v.document_schema()
        second, hit2, _ = self.client.call('/completions', body, cache=True)
        final = api.ordered_choices(second, 1)[0]
        obj = v.document_parse(final['text']) if final['finish_reason'] == 'stop' else None
        return dict(key=job['key'], doc_id=job['doc_id'], pass_id=job['pass_id'], budget=job['budget'],
                    status='valid' if obj else 'invalid_answer', answer=final['text'],
                    thinking_forced=trace.get('stop_reason') != '</think>', thinking_finish=trace['finish_reason'],
                    answer_finish=final['finish_reason'], prompt_tokens=len(before),
                    think_tokens=len(trace.get('token_ids') or self.tok.encode(trace['text'], add_special_tokens=False)),
                    answer_tokens=len(final.get('token_ids') or []), seconds=time.monotonic() - t0,
                    cache_hits=int(hit1) + int(hit2), host=os.uname().nodename)


def shard_jobs(jobs, shard, shards):
    """Documents round-robin by id order so every shard gets a similar length mix; long reads first."""
    doc_ids = sorted({j['doc_id'] for j in jobs})
    mine = {d for i, d in enumerate(doc_ids) if i % shards == shard}
    picked = [j for j in jobs if j['doc_id'] in mine]
    return sorted(picked, key=lambda j: (-j['thinking'], j['doc_id'], j['pass_id']))


def run(out: Path, url: str, shard: int, shards: int, concurrency: int):
    jobs = read_jsonl(out / 'jobs.jsonl')
    assert jobs and 0 <= shard < shards and concurrency > 0
    done = {r['key'] for p in out.glob('responses-*.jsonl') for r in read_jsonl(p)}
    pending = [j for j in shard_jobs(jobs, shard, shards) if j['key'] not in done]
    print('MP_SHARD', api.canonical(dict(shard=shard, shards=shards, pending=len(pending),
                                          concurrency=concurrency, url=url)), flush=True)
    reader = Reader(url, out / 'http-cache')
    lock, sink = threading.Lock(), (out / f'responses-{shard}.jsonl').open('a', buffering=1)
    stats, t0 = Counter(), time.monotonic()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(reader.read, job): job for job in pending}
        for future in as_completed(futures):
            try:
                row = future.result()
            except Exception as exc:  # keep the shard going; the read stays pending for a resume
                stats['errors'] += 1
                print('MP_ERROR', futures[future]['key'], repr(exc)[:300], flush=True)
                continue
            with lock:
                sink.write(api.canonical(row) + '\n')
            stats['done'] += 1
            stats['tokens'] += row['think_tokens'] + row['answer_tokens']
            if stats['done'] % 10 == 0 or stats['done'] == len(pending):
                el = time.monotonic() - t0
                print('MP_PROGRESS', api.canonical(dict(shard=shard, done=stats['done'], total=len(pending),
                      errors=stats['errors'], gen_tok_s=round(stats['tokens'] / el, 1),
                      elapsed_min=round(el / 60, 1))), flush=True)
    sink.close()
    return 0 if stats['errors'] == 0 else 1


def reading_pairs(row, universe, quality):
    obj = v.document_parse(row.get('answer')) if row and row['status'] == 'valid' else None
    if obj is None:
        quality['invalid_reads'] += 1
        return set()
    chosen = set()
    for item in obj['relations']:
        pair = tuple(sorted(x.strip() for x in item['pair'].split(',')))
        if pair in universe:
            chosen.add(pair)
        else:
            quality['out_of_universe_items'] += 1
    return chosen


def vote(reads, threshold):
    return {p for p in set().union(*reads) if sum(p in r for r in reads) >= threshold} if reads else set()


def per_doc_counts(pred, docs):
    out = {}
    for doc_id, chosen in pred.items():
        gold, universe = f.gold_and_universe('MAVEN', docs[doc_id])
        out[doc_id] = c.counts(a.symmetric(chosen), gold, universe)
    return out


def systems_from(jobs, rows, docs):
    """Missing or invalid reads count empty: the 5/10-read denominators stay fixed."""
    reads = defaultdict(lambda: defaultdict(list))
    quality = Counter()
    for job in sorted(jobs, key=lambda j: (j['doc_id'], j['budget'], j['pass_id'])):
        row = rows.get(job['key'])
        if row is None:
            quality['missing_reads'] += 1
        universe = set(c.pairs(docs[job['doc_id']]))
        reads[job['doc_id']][job['budget']].append(reading_pairs(row, universe, quality))
    systems = defaultdict(dict)
    for doc_id, by in reads.items():
        short, long_ = by['short'], by['long']
        assert len(short) == 5 and len(long_) == 5
        systems['short_vote5'][doc_id] = vote(short, 3)
        systems['long_vote5'][doc_id] = vote(long_, 3)
        systems['pool_vote10_ge6'][doc_id] = vote(short + long_, 6)
        systems['pool_vote10_ge5'][doc_id] = vote(short + long_, 5)
        for i in range(5):
            systems[f'short_pass{i}'][doc_id] = short[i]
            systems[f'long_pass{i}'][doc_id] = long_[i]
    return systems, quality


def score(out: Path):
    docs = json.loads((out / 'documents.json').read_text())
    jobs = read_jsonl(out / 'jobs.jsonl')
    rows = {r['key']: r for p in sorted(out.glob('responses-*.jsonl')) for r in read_jsonl(p)}
    systems, quality = systems_from(jobs, rows, docs)
    counts = {name: per_doc_counts(pred, docs) for name, pred in systems.items()}
    metrics = {name: c.metric(per) for name, per in counts.items()}
    deltas = {f'{x}-vs-{y}': paired_delta(counts[x], counts[y], draws=5000, seed=SEED)
              for x, y in (('pool_vote10_ge6', 'short_vote5'), ('pool_vote10_ge6', 'long_vote5'),
                           ('long_vote5', 'short_vote5'), ('pool_vote10_ge5', 'pool_vote10_ge6'))}
    valid = [r for r in rows.values()]
    forced = {b: sum(r['thinking_forced'] for r in valid if r['budget'] == b) for b in BUDGETS}
    tokens = {b: sum(r['think_tokens'] for r in valid if r['budget'] == b) for b in BUDGETS}
    report = dict(protocol=VERSION, documents=len(docs), reads_expected=len(jobs), reads_done=len(rows),
                  quality=dict(quality), forced_closes=forced, think_tokens=tokens,
                  metrics=metrics, paired_deltas=deltas, per_document=counts['pool_vote10_ge6'])
    (out / 'report.json').write_text(json.dumps(report, indent=1))
    for name in ('short_vote5', 'long_vote5', 'pool_vote10_ge6', 'pool_vote10_ge5'):
        m = metrics[name]
        print(f"MP_METRIC {name}: F1 {m['f1']:.2f} P {m['precision']:.2f} R {m['recall']:.2f} "
              f"TP {m['tp']} FP {m['fp']} FN {m['fn']}", flush=True)
    for name, (d, lo, hi) in deltas.items():
        print(f'MP_DELTA {name}: {d:+.2f} [{lo:+.2f}, {hi:+.2f}]', flush=True)
    print('MP_QUALITY', api.canonical(dict(quality=dict(quality), forced=forced, reads=len(rows))), flush=True)
    return 0


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('cmd', choices=('prepare', 'run', 'score'))
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--url', default='http://127.0.0.1:8000/v1')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--concurrency', type=int, default=48)
    args = ap.parse_args(argv)
    if args.cmd == 'prepare':
        prepare(args.out)
        return 0
    if args.cmd == 'run':
        return run(args.out, args.url, args.shard, args.shards, args.concurrency)
    return score(args.out)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
