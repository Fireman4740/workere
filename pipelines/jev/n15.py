"""Small tracked local-only generation-contract pilot; never retries inference."""
import collections
import json
import os
import pathlib
import random
import sys
import time
from pipelines.jev import n13 as h

P = pathlib.Path('experiments/jev/n15')


def log(kind, value):
    print('N15_' + kind + ' ' + json.dumps(value), flush=True)


def preflight():
    spec = h.read(P / 'protocol.json')
    raw = (P / 'requests.jsonl').read_bytes()
    assert h.sha(raw) == spec['input_sha256']
    rows = [json.loads(line) for line in raw.splitlines()]
    docs = {d['id']: d for d in h.read(P / 'public.json')}
    bank = {d['id']: d for d in h.read(P / 'demo-bank.json')['documents']}
    assert len(rows) == len({r['logical_id'] for r in rows}) == 36
    assert set(docs) == set(spec['documents']) and len(docs) == 6
    assert not set(docs).intersection(bank)
    for doc in docs:
        for slot in range(3):
            pair = [r for r in rows if r['doc'] == doc and r['slot'] == slot]
            assert {r['condition'] for r in pair} == {'medium', 'no_thinking'} and len(pair) == 2
            assert pair[0]['body']['messages'] == pair[1]['body']['messages']
            assert pair[0]['demo'] == pair[1]['demo']
    for r in rows:
        b = r['body']
        assert h.sha(h.canonical(b)) == r['body_sha256']
        assert b['model'] == spec['model'] and b['max_tokens'] == spec['max_tokens'] == 8192
        assert b['temperature'] == 0 and not b['stream']
        assert bank[r['demo']]['fold'] != int(docs[r['doc']]['idx']) % 10
        expected = {'enable_thinking': r['condition'] == 'medium', 'preserve_thinking': True}
        if r['condition'] == 'medium':
            expected['reasoning_effort'] = 'medium'
            assert b['reasoning_effort'] == 'medium'
        else:
            assert 'reasoning_effort' not in b
        assert b['chat_template_kwargs'] == expected
    return spec, rows, docs


def freeze(rows, events, docs):
    assert len(events) == len(rows) == 36, 'incomplete pilot: no primary score'
    by = {e['logical_id']: e for e in events}
    assert len(by) == len(rows) and not any(e['error'] for e in events)
    assert len({e['response']['id'] for e in events}) == len(events)
    votes = collections.Counter()
    for r in rows:
        e = by[r['logical_id']]
        assert e['request'] == r['body']
        positives = h.verify_response(e['response'], set(map(tuple, docs[r['doc']]['pair_list'])))
        for pair in positives:
            votes[r['condition'], r['doc'], pair] += 1
    return [dict(condition=c, doc=d, a=a, b=b,
                 votes=votes[c, d, (a, b)], positive=votes[c, d, (a, b)] >= 2)
            for c in ('medium', 'no_thinking') for d in sorted(docs)
            for a, b in sorted({tuple(sorted(p)) for p in docs[d]['pair_list']})]


def score(decisions, gold):
    truth = {(r['doc'], r['a'], r['b']): bool(r['gold']) for r in gold}
    result = {}
    for c in ('medium', 'no_thinking'):
        rs = [r for r in decisions if r['condition'] == c]
        assert len(rs) == len(truth)
        assert {(r['doc'], r['a'], r['b']) for r in rs} == set(truth)
        per = collections.defaultdict(lambda: [0, 0, 0])
        for r in rs:
            y = truth[r['doc'], r['a'], r['b']]; p = r['positive']
            per[r['doc']][0] += int(p and y)
            per[r['doc']][1] += int(p and not y)
            per[r['doc']][2] += int(not p and y)
        tp, fp, fn = [sum(v[i] for v in per.values()) for i in range(3)]
        result[c] = dict(TP=tp, FP=fp, FN=fn, F1=200*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,
                         per_document_counts=dict(per))
    return result


def main():
    spec, rows, docs = preflight()
    if '--preflight' in sys.argv:
        log('PREFLIGHT', dict(status='PASS', documents=len(docs), calls=len(rows))); return
    state = pathlib.Path(spec['state_root']).expanduser() / os.environ['ORX_RUN_ID']
    assert not state.exists(), 'fresh tracked run required'
    state.mkdir(parents=True)
    for name in ('protocol.json', 'requests.jsonl', 'public.json', 'demo-bank.json', 'historical-context.json'):
        (state/name).write_bytes((P/name).read_bytes())
    events = []; started = time.monotonic()
    try:
        http = h.LocalHTTP(h.credential(spec))
        h.ready_model(http, spec, state)
        counts = {}; probes = {}
        for r in rows:
            tokenized = h.tokenize(http, spec, r)
            counts[r['logical_id']] = tokenized['prompt_tokens_exact_service']
            probes[r['logical_id']] = tokenized['tokenizer_response']
        # Verify the switch actually changes the template for identical messages.
        for d in docs:
            for slot in range(3):
                a = probes[f'n15:{d}:{slot}:medium'].get('tokens')
                b = probes[f'n15:{d}:{slot}:no_thinking'].get('tokens')
                assert a and b and a != b, 'thinking switch not observable in tokenizer'
        h.save(state/'tokenizer-preflight.json', dict(counts=counts, responses=probes))
        log('CONFIG', spec)
        # Interleave conditions within each paired doc/example; no concurrent GPU jobs.
        groups = [(d, s) for d in sorted(docs) for s in range(3)]
        rng = random.Random(2026110315); rng.shuffle(groups)
        order = []
        for d, s in groups:
            pair = [r for r in rows if r['doc'] == d and r['slot'] == s]
            rng.shuffle(pair); order.extend(pair)
        h.save(state/'submission-order.json', [r['logical_id'] for r in order])
        for r in order:
            remaining = spec['wall_seconds'] - (time.monotonic() - started)
            if remaining <= 0: raise RuntimeError('pilot wall budget exhausted')
            response = None; error = None; t = time.monotonic()
            try:
                response = http('POST', spec['base_url']+'/chat/completions', r['body'],
                                timeout=min(spec['request_timeout_seconds'], remaining))
                h.verify_response(response, set(map(tuple, docs[r['doc']]['pair_list'])))
                u = response['usage']
                assert u['prompt_tokens'] == counts[r['logical_id']]
                assert 0 <= u['completion_tokens'] <= spec['max_tokens']
            except Exception as exc:
                error = dict(type=type(exc).__name__, message=str(exc)[:500])
            e = dict(logical_id=r['logical_id'], condition=r['condition'], request=r['body'],
                     response=response, error=error, seconds=time.monotonic()-t)
            events.append(e)
            with (state/'events.jsonl').open('a') as f: f.write(json.dumps(e)+'\n')
            log('PROGRESS', dict(completed=len(events), maximum=36, condition=r['condition'],
                                 seconds=e['seconds'], error=error))
            if error: raise RuntimeError('invalid response; stopped without retry or negative imputation')
        decisions = freeze(rows, events, docs)
        h.save(state/'decisions-before-gold.json', decisions)
        h.save(state/'freeze.json', dict(decisions_sha256=h.sha(h.canonical(decisions)), gold_opened=False))
        gold = h.read(P/'gold.json'); h.save(state/'gold-evaluated.json', gold)
        result = dict(status='COMPLETE_DIAGNOSTIC', systems=score(decisions, gold), promotion=False,
                      wall_seconds=time.monotonic()-started, actual_calls=len(events))
        result['resources'] = {c:dict(seconds=sum(e['seconds'] for e in events if e['condition']==c),
            completion_tokens=sum(e['response']['usage']['completion_tokens'] for e in events if e['condition']==c))
            for c in spec['conditions']}
        h.save(state/'result.json', result); log('RESULT', result)
    except Exception as exc:
        result = dict(status='BLOCKED_VALIDITY', actual_calls=len(events), metrics=None,
                      error=type(exc).__name__, message=str(exc)[:500])
        h.save(state/'result.json', result); log('RESULT', result)
        raise
    finally:
        h.export(state, 'N15_FINAL')


if __name__ == '__main__':
    main()
