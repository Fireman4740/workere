#!/usr/bin/env python3
"""Tracked offline gate / bounded live M0. Run only via the fixed experiment entry.
No synchronous fallback; no paid retry; durable ambiguous-submission receipts.
"""
import base64
from collections import Counter, defaultdict
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eci_m0_pilot.m0 import as_pairs, build, validate
from eci_m0_pilot.protocol import ARMS, coverage, evaluate, json_text, ledger_for, requests_for

HERE = Path(__file__).resolve().parent
ENDPOINT = 'https://openrouter.ai/api/v1/batches'
MODELS = 'https://openrouter.ai/api/v1/models'

def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json_text(obj) + '\n'); temp.replace(path)

def emit(name, obj):
    print(name + ' ' + json_text(obj), flush=True)

def export(files):
    data = gzip.compress(json_text(files).encode(), mtime=0)
    encoded = base64.b64encode(data).decode()
    sha = hashlib.sha256(data).hexdigest()
    for i in range(0, len(encoded), 12000):
        emit('M0_EXPORT_CHUNK', {'sha256': sha, 'index': i // 12000, 'data': encoded[i:i+12000]})
    emit('M0_EXPORT_END', {'sha256': sha, 'chunks': math.ceil(len(encoded)/12000), 'files': sorted(files)})

def http(method, url, payload=None):
    key = os.environ.get('OPENROUTER_API_KEY') or os.environ.get('OPENROUTER')
    if not key: raise RuntimeError('missing OPENROUTER_API_KEY / OPENROUTER; no request sent')
    req = urllib.request.Request(url, method=method,
        data=json.dumps(payload, ensure_ascii=False).encode() if payload is not None else None,
        headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=120) as res: return json.load(res)
    except urllib.error.HTTPError as e:
        # Never emit request headers or provider bodies that might echo credentials.
        raise RuntimeError(f'provider HTTP {e.code} on {method}; no automatic paid retry') from None

def catalog_gate(settings, required_context):
    data = http('GET', MODELS)
    model_id = settings['model'].removesuffix(':batch')
    models = [m for m in data.get('data', []) if m.get('id') == model_id]
    if len(models) != 1: raise RuntimeError('frozen model missing/ambiguous in provider catalog; no substitution')
    model = models[0]; pricing = model.get('pricing') or {}
    prices = {k: float(pricing[k]) * 1e6 for k in ('prompt', 'completion')}
    if any(not math.isfinite(p) or p < 0 for p in prices.values()): raise RuntimeError('invalid provider prices')
    # Catalog is a conservative ceiling; do not assume an undocumented batch discount.
    if prices['prompt'] > settings['input_price_per_million'] or prices['completion'] > settings['output_price_per_million']:
        raise RuntimeError('catalog price exceeds frozen reservation ceiling; no submission')
    if model.get('context_length', 0) < required_context:
        raise RuntimeError('catalog context shorter than required conservative prompt+output bound; no truncation')
    return {'id': model_id, 'context_length': model.get('context_length'), 'catalog_prices_per_million': prices,
            'note': 'catalog identity is not proof of immutable backend weights; served identity is separately audited'}

def batch_run(rows, settings, state):
    payload = {'endpoint': '/v1/chat/completions', 'model': settings['model'].removesuffix(':batch'),
               'completion_window': '24h',
               'requests': [{'custom_id': r['custom_id'], 'body': r['body']} for r in rows]}
    fingerprint = hashlib.sha256(json_text(payload).encode()).hexdigest()
    receipt = state / (fingerprint + '.json')
    state.mkdir(parents=True, exist_ok=True)
    reserve = len(rows) * settings['reservation_per_request_usd']
    if len(rows) > settings['max_logical_requests'] or reserve > settings['budget_usd']:
        raise RuntimeError('frozen call or monetary cap exceeded; nothing sent')
    with receipt.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if receipt.exists():
            record = json.loads(receipt.read_text())
            if not record.get('batch', {}).get('id'):
                raise RuntimeError('submission outcome unknown/rejected; receipt prevents duplicate paid POST')
            emit('M0_BATCH_RESUME', {'batch_id': record['batch']['id'], 'receipt': str(receipt)})
        else:
            record = {'state': 'SUBMITTING', 'request_sha256': fingerprint, 'reservation_usd': reserve,
                      'logical_requests': len(rows), 'first_run_id': os.environ.get('ORX_RUN_ID'), 'payload': payload}
            save(receipt, record)  # written BEFORE the only paid POST
            try:
                batch = http('POST', ENDPOINT, payload)
                if not isinstance(batch.get('id'), str): raise RuntimeError('provider did not return a batch id')
            except BaseException:
                emit('M0_SUBMISSION_UNCERTAIN', {'receipt': str(receipt), 'reservation_usd': reserve})
                raise
            record.update(state='ACCEPTED', batch=batch); save(receipt, record)
            emit('M0_BATCH_SUBMITTED', {'batch_id': batch['id'], 'requests': len(rows),
                                      'reservation_usd': reserve, 'receipt': str(receipt)})
    deadline = time.monotonic() + settings['poll_deadline_seconds']
    while record['batch'].get('status') not in {'completed', 'failed', 'expired', 'cancelled'}:
        if time.monotonic() >= deadline: raise RuntimeError('poll timeout; receipt retained; do not resubmit')
        emit('M0_BATCH_PROGRESS', {k: record['batch'].get(k) for k in ('id', 'status', 'request_counts')})
        time.sleep(30)
        try: updated = http('GET', ENDPOINT + '/' + record['batch']['id'])
        except (RuntimeError, urllib.error.URLError, TimeoutError): continue
        record['batch'] = updated; save(receipt, record)
    batch = record['batch']
    emit('M0_BATCH_TERMINAL', {k: batch.get(k) for k in ('id', 'status', 'request_counts', 'usage')})
    if batch.get('status') != 'completed': raise RuntimeError('batch terminal failure; no paid retry or fallback')
    items = batch.get('results') or []
    by_id = {x.get('custom_id'): x for x in items}
    if len(by_id) != len(items) or set(by_id) != {r['custom_id'] for r in rows}:
        raise RuntimeError('batch results do not exactly cover requests; retain receipt for audit')
    cost = (batch.get('usage') or {}).get('cost')
    if cost is not None and (type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0):
        raise RuntimeError('invalid reported batch cost')
    return by_id, {'batch_id': batch['id'], 'provider_reported_total_cost_usd': cost,
                  'reservation_usd': reserve, 'within_reservation': cost is None or cost <= reserve,
                  'logical_requests': len(rows), 'paid_post_attempts_for_payload': 1,
                  'cost_unreported_reservation_retained': cost is None, 'first_run_id': record['first_run_id']}

def load_data():
    data = HERE / 'data'
    provenance = json.loads((data / 'provenance.json').read_text())
    for name, sha in provenance['files_sha256'].items():
        assert hashlib.sha256((data / name).read_bytes()).hexdigest() == sha, f'data mismatch: {name}'
    docs = [json.loads(s) for s in (data / 'public_train233.jsonl').read_text().splitlines()]
    assert len({d['doc_id'] for d in docs}) == len(docs)
    for d in docs:
        assert set(d) == {'doc_id', 'topic', 'text', 'mentions', 'pairs'}
    return docs, json.loads((data / 'selection.json').read_text()), json.loads((data / 'demonstrations.json').read_text()), json.loads((data / 'prompt.json').read_text())

def run():
    cfg = json.loads((HERE / 'config.json').read_text())
    docs, selection, demos, template = load_data()
    doc_by_id = {d['doc_id']: d for d in docs}
    ids = selection['smoke_document_ids'] if cfg['subset'] == 'smoke' else selection['document_ids']
    emit('M0_CONFIG', {'config': cfg, 'document_ids': ids, 'commit': os.environ.get('ORX_COMMIT'),
         'run_id': os.environ.get('ORX_RUN_ID'), 'selection_uses_labels': False,
         'reference_preserved': True, 'new_model_calls_at_start': 0})
    out = ROOT / 'logs' / ('m0-' + os.environ.get('ORX_RUN_ID', 'lightweight-check'))
    summary = []
    for doc in docs:
        rows = build(doc, 7, cfg['seed']); validate(rows, doc, 7)
        blocks = [r for r in rows if r['arm'] == 'M']
        sizes = [len(r['allowed_original']) for r in blocks]
        summary.append({'doc_id': doc['doc_id'], 'pairs': len(as_pairs(doc['pairs'])),
                        'block_sizes': sizes, 'total_size_range': max(sizes)-min(sizes)})
    prepared = {did: requests_for(doc_by_id[did], demos, selection['demonstration_schedule'], template, cfg) for did in ids}
    allrows = [r for did in ids for r in prepared[did]]
    assert len({r['custom_id'] for r in allrows}) == len(allrows)
    by_arm = {}
    for arm in ARMS:
        rr = [r for r in allrows if r['arm'] == arm]
        by_arm[arm] = {'requests': len(rr), 'prompt_characters': sum(r['prompt_characters'] for r in rr),
                       'prompt_utf8_upper_bound': sum(r['prompt_utf8_upper_bound'] for r in rr),
                       'largest_prompt_upper_bound': max(r['prompt_utf8_upper_bound'] for r in rr)}
    offline = {'documents': len(docs), 'candidate_pairs': sum(x['pairs'] for x in summary),
               'all_pairs_covered_once_in_M_and_MP': True, 'planned_requests': len(allrows),
               'per_arm_preflight': by_arm, 'max_total_block_size_range': max(x['total_size_range'] for x in summary),
               'planner_note': 'supplied independently rotated stratum quotas balance each stratum within one, not necessarily total within one',
               'token_count_status': 'UTF8+1024 conservative proxy, not exact tokenizer tokens',
               'selected_documents': ids, 'new_model_calls': 0}
    save(out / 'offline.json', offline); save(out / 'plan-summary-train233.json', summary)
    save(out / 'requests.json', prepared)
    emit('M0_OFFLINE_SUMMARY', offline)
    files = {'offline.json': offline, 'plan-summary-train233.json': summary,
             'selection.json': selection, 'requests.json': prepared}
    if cfg['stage'] == 'offline':
        export(files); emit('M0_RESULT', {'stage': 'offline', 'status': 'PASS', 'new_model_calls': 0,
                                        'f1_measured': False}); return
    if cfg['stage'] != 'live': raise ValueError('unknown stage')
    gate = catalog_gate(cfg, max(r['prompt_utf8_upper_bound'] for r in allrows) + cfg['max_output_tokens']); emit('M0_PROVIDER_GATE', gate)
    state = Path('~/.local/share/openresearch/workere-decisions/m0-coverage-20261027').expanduser()
    results, cost = batch_run(allrows, cfg, state)
    ledgers, events, coverage_out = {}, [], {}
    for did in ids:
        ledger, ev = ledger_for(doc_by_id[did], prepared[did], results, cfg)
        ledgers[did] = ledger; events.extend(ev)
        coverage_out[did] = {arm: coverage(doc_by_id[did], ledger[arm]) for arm in ARMS}
    # Freeze all outputs before opening evaluation-only labels.
    frozen = {'ledgers': ledgers, 'events': events, 'coverage': coverage_out, 'batch_cost': cost, 'raw_results': results}
    save(out / 'outputs-before-gold.json', frozen)
    output_hash = hashlib.sha256((out / 'outputs-before-gold.json').read_bytes()).hexdigest()
    emit('M0_OUTPUTS_FROZEN', {'sha256': output_hash, 'gold_opened': False})
    evaluation = [r for r in json.loads((HERE / 'data/evaluation_only.json').read_text()) if r['doc_id'] in ids]
    report = evaluate(evaluation, ledgers)
    report['cost'] = cost
    report['events'] = dict(Counter(e['state'] for e in events))
    report['outputs_before_gold_sha256'] = output_hash
    report['provider_gate'] = gate
    report['cost_by_arm'] = {}
    for arm in ARMS:
        aa = [e for e in events if e['arm'] == arm]
        pp = sum(e.get('usage', {}).get('prompt_tokens', 0) or 0 for e in aa)
        cc = sum(e.get('usage', {}).get('completion_tokens', 0) or 0 for e in aa)
        reasoning = [e.get('usage', {}).get('completion_tokens_details', {}).get('reasoning_tokens') for e in aa]
        report['cost_by_arm'][arm] = {'prompt_tokens': pp, 'completion_tokens_including_reasoning_if_provider_includes_it': cc,
                 'reasoning_tokens': sum(x for x in reasoning if x is not None) if any(x is not None for x in reasoning) else None,
                 'usage_available_responses': sum(bool(e.get('usage')) for e in aa), 'responses': len(aa),
                 'weighted_token_proxy_usd': (pp*cfg['input_price_per_million']+cc*cfg['output_price_per_million'])/1e6,
                 'cache_details': [e.get('usage', {}).get('prompt_tokens_details') for e in aa]}
    o = report['cost_by_arm']['O']['weighted_token_proxy_usd']; m = report['cost_by_arm']['M']['weighted_token_proxy_usd']
    complete_usage = all(x['usage_available_responses'] == x['responses'] for x in report['cost_by_arm'].values())
    report['M_O_observable_token_cost_within_5pct'] = complete_usage and o > 0 and abs(m/o-1) <= .05
    report['equal_calls_is_not_equal_compute'] = True
    report['followup_policy'] = 'No automatic extension: inspect technical validity, additions and O comparison; stop for FP inflation or vanishing cost advantage. E1/E2 require separate frozen experiments.'
    save(out / 'results.json', report)
    files.update({'results.json': report, 'outputs-before-gold.json': frozen})
    export(files)
    emit('M0_RESULT', report)
    if any(e['state'] == 'MODEL_MISMATCH' for e in events): raise RuntimeError('model mismatch: results not promotable')
    if not cost['within_reservation']: raise RuntimeError('reported cost exceeds frozen reservation: stop')

if __name__ == '__main__': run()
