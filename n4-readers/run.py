"""N4 paired document readers, frozen inputs, no training or human annotation."""
import base64
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import urllib.error
import urllib.request

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from eci_m0_pilot.protocol import decode_response

ENDPOINT = 'https://openrouter.ai/api/v1/batches'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, allow_nan=False))
    temp.replace(path)


def emit(tag, obj):
    print(tag + ' ' + json.dumps(obj, ensure_ascii=False, sort_keys=True, allow_nan=False), flush=True)


def http(method, url, payload=None):
    key = os.environ.get('OPENROUTER_API_KEY') or os.environ.get('OPENROUTER')
    if not key:
        raise RuntimeError('missing provider credential; no request sent')
    req = urllib.request.Request(url, method=method,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError('provider HTTP %s; no paid retry' % exc.code) from None


def export(out):
    files = {p.name: p.read_text() for p in sorted(out.iterdir()) if p.is_file()}
    data = gzip.compress(json.dumps(files, sort_keys=True).encode(), mtime=0)
    encoded = base64.b64encode(data).decode()
    for index in range(0, len(encoded), 12000):
        emit('N4_EXPORT_CHUNK', {'sha256': digest(data), 'index': index // 12000,
                                'data': encoded[index:index + 12000]})
    emit('N4_EXPORT_END', {'sha256': digest(data), 'chunks': math.ceil(len(encoded) / 12000),
                          'files': sorted(files)})


def counts(pred, gold):
    tp, fp, fn = len(pred & gold), len(pred - gold), len(gold - pred)
    return {'tp': tp, 'fp': fp, 'fn': fn, 'f1': 2 * tp / (2 * tp + fp + fn) if tp else 0}


def bootstrap(a, b, gold, universe, repetitions=4000):
    topics = sorted({k[0].split('_')[0] for k in universe})
    group = {t: {k for k in universe if k[0].split('_')[0] == t} for t in topics}
    per = {t: (counts(a & keys, gold & keys), counts(b & keys, gold & keys)) for t, keys in group.items()}
    rng, deltas = random.Random(20261028), []
    for _ in range(repetitions):
        draw = [rng.choice(topics) for _ in topics]
        def total(side):
            c = [sum(per[t][side][k] for t in draw) for k in ['tp', 'fp', 'fn']]
            return 2 * c[0] / (2 * c[0] + c[1] + c[2]) if c[0] else 0
        deltas.append(total(0) - total(1))
    deltas.sort()
    return {'delta': counts(a, gold)['f1'] - counts(b, gold)['f1'],
            'ci95': [deltas[int(.025 * repetitions)], deltas[int(.975 * repetitions) - 1]],
            'unit': 'topic', 'topics': len(topics), 'repetitions': repetitions,
            'caution': 'One fresh three-pass realization per reader; exposed development'}


def evaluate(public, events, labels):
    universe = {(r['doc_id'], *sorted(r['pair'])) for r in labels}
    gold = {(r['doc_id'], *sorted(r['pair'])) for r in labels if r['gold']}
    votes = {arm: Counter() for arm in ['Luna', 'DeepSeek']}
    for event in events:
        assert event['valid']
        votes[event['arm']].update((event['doc_id'], *sorted(pair)) for pair in event['positive'])
    predictions = {'P0': set()}
    for arm in votes:
        predictions[arm + '_majority'] = {k for k, value in votes[arm].items() if value >= 2}
        predictions[arm + '_fusion'] = set()
    for did, reference in public['reference'].items():
        for row in reference:
            key = (did, *sorted(row['pair']))
            assert key in universe
            if row['baseline_positive']:
                predictions['P0'].add(key)
            d1 = sum(o['d1'] for o in row['orientations']) / len(row['orientations'])
            for arm in votes:
                if (votes[arm][key] / 3 + d1) / 2 >= .39:
                    predictions[arm + '_fusion'].add(key)
    assert all(p <= universe for p in predictions.values())
    return {'metrics': {name: counts(p, gold) for name, p in predictions.items()},
            'reader_contrast': bootstrap(predictions['DeepSeek_majority'], predictions['Luna_majority'], gold, universe),
            'fusion_contrast': bootstrap(predictions['DeepSeek_fusion'], predictions['Luna_fusion'], gold, universe),
            'candidates': len(universe), 'gold_positives': len(gold),
            'predictions': {name: sorted(p) for name, p in predictions.items()},
            'promotion': False, 'scope': 'exploratory development, not confirmation or equal realized cost'}


def main():
    run_id = os.environ.get('ORX_RUN_ID')
    if not run_id:
        raise RuntimeError('Tracked launch required')
    frozen = HERE / 'frozen'
    protocol = json.loads((frozen / 'protocol.json').read_text())
    raw_public = (frozen / 'public.json').read_bytes()
    assert digest(raw_public) == protocol['public_sha256']
    public = json.loads(raw_public)
    rows = public['requests']
    assert len(rows) == len({r['custom_id'] for r in rows}) == 72
    for did in protocol['documents']:
        for slot in [1, 2, 3]:
            rr = [r for r in rows if r['doc_id'] == did and r['slot'] == slot]
            assert len(rr) == 2 and rr[0]['body']['messages'] == rr[1]['body']['messages']
    out = ROOT / 'logs' / ('n4-' + run_id)
    out.mkdir(parents=True, exist_ok=True)
    save(out / 'protocol.json', protocol)
    emit('N4_CONFIG', {'run_id': run_id, 'commit': os.environ.get('ORX_COMMIT'), 'protocol': protocol})
    catalogue = http('GET', 'https://openrouter.ai/api/v1/models')['data']
    for model in protocol['models'].values():
        item = next(x for x in catalogue if x['id'] == model['id'].removesuffix(':batch'))
        assert item['canonical_slug'] == model['canonical']
        assert float(item['pricing']['prompt']) * 1e6 <= model['input'] + 1e-10
        assert float(item['pricing']['completion']) * 1e6 <= model['output'] + 1e-10
        assert item['context_length'] >= max(len(json.dumps(r['body']).encode()) + 1024 + 32768 for r in rows)
    ceiling = sum(((len(json.dumps(r['body'], ensure_ascii=False).encode()) + 1024) * protocol['models'][r['arm']]['input']
                   + 32768 * protocol['models'][r['arm']]['output']) / 1e6 for r in rows)
    assert ceiling <= protocol['budget_usd'] and abs(ceiling - protocol['api_ceiling_usd']) < 1e-10
    state = Path('~/.local/share/openresearch/workere-decisions/campaign-20261028/n4').expanduser() / digest((frozen / 'protocol.json').read_bytes())
    state.mkdir(parents=True, exist_ok=True)
    with (state / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save(state / 'reservation.json', {'maximum_usd': ceiling, 'cap_usd': protocol['budget_usd']})
        # The only previous submission was definitively rejected HTTP422, no batch id.
        # Preserve its receipt and explicitly release its unspent reservation.
        prior_hash = protocol.get('prior_rejected_protocol_sha256')
        if prior_hash:
            prior = state.parent / prior_hash
            prior_receipt = prior / 'DeepSeek.json'
            old = json.loads(prior_receipt.read_text())
            assert old.get('first_run_id') == protocol['prior_rejected_run'] and not old.get('batch')
            old['resolution'] = {'status': 'REJECTED_HTTP422', 'evidence_run': protocol['prior_rejected_run'],
                                 'accepted_batch_id': None, 'repair_run': run_id, 'reserved_budget_released': True}
            save(prior_receipt, old)
            emit('N4_TECHNICAL_REPAIR', old['resolution'])
        batches = {}
        # Submit Luna's asynchronous batch first, then use the supported direct
        # transport for DeepSeek. Messages, decoding and logical calls unchanged.
        for arm, model in protocol['models'].items():
            if protocol['transport'][arm] != 'batch':
                continue
            rr = [r for r in rows if r['arm'] == arm]
            payload = {'endpoint': '/v1/chat/completions', 'model': model['id'].removesuffix(':batch'),
                       'completion_window': '24h', 'requests': [{'custom_id': r['custom_id'],
                       'body': {k: v for k, v in r['body'].items() if k != 'model'}} for r in rr]}
            receipt = state / (arm + '.json')
            if receipt.exists():
                record = json.loads(receipt.read_text())
                assert record['payload_sha256'] == digest(json.dumps(payload).encode())
                if not record.get('batch', {}).get('id'):
                    raise RuntimeError('Uncertain prior POST, not resubmitting')
            else:
                record = {'status': 'SUBMITTING', 'first_run_id': run_id,
                          'payload_sha256': digest(json.dumps(payload).encode()), 'payload': payload}
                save(receipt, record)
                record['batch'] = http('POST', ENDPOINT, payload)
                assert record['batch'].get('id')
                save(receipt, record)
            batches[arm] = record
            emit('N4_BATCH_SUBMITTED', {'arm': arm, 'id': record['batch']['id'], 'requests': len(rr), 'ceiling_total_usd': ceiling})
        def direct(row):
            receipt = state / ('direct-' + row['custom_id'] + '.json')
            if receipt.exists():
                record = json.loads(receipt.read_text())
                if record.get('status') != 'DELIVERED':
                    raise RuntimeError('Uncertain prior direct POST; never resubmitting')
                return record['item']
            record = {'status': 'SUBMITTING', 'first_run_id': run_id, 'request': row['body']}
            save(receipt, record)
            try:
                body = dict(row['body'])
                body['model'] = body['model'].removesuffix(':batch')
                response = http('POST', 'https://openrouter.ai/api/v1/chat/completions', body)
                item = {'custom_id': row['custom_id'], 'response': {'status_code': 200, 'body': response}}
            except Exception as exc:
                item = {'custom_id': row['custom_id'], 'error': {'type': type(exc).__name__, 'message': str(exc)}}
            record.update(status='DELIVERED', item=item)
            save(receipt, record)
            emit('N4_DIRECT_DELIVERED', {'custom_id': row['custom_id'], 'error': bool(item.get('error'))})
            return item
        direct_rows = [r for r in rows if protocol['transport'][r['arm']] == 'direct']
        with ThreadPoolExecutor(max_workers=4) as workers:
            direct_results = list(workers.map(direct, direct_rows))
        direct_costs = [((r.get('response') or {}).get('body') or {}).get('usage', {}).get('cost') for r in direct_results]
        direct_cost = sum(direct_costs) if all(type(v) in (int, float) for v in direct_costs) else None
        batches['DeepSeek'] = {'transport': 'direct', 'batch': {'id': 'direct-completions', 'status': 'completed',
            'results': direct_results, 'usage': {'cost': direct_cost},
            'request_counts': {'total': len(direct_rows), 'failed': sum(bool(r.get('error')) for r in direct_results)}}}
        deadline = time.monotonic() + 26 * 3600
        while any(r['batch'].get('status') not in {'completed', 'failed', 'expired', 'cancelled'} for r in batches.values()):
            if time.monotonic() > deadline:
                raise RuntimeError('GET deadline; preserve receipts; no resubmission')
            for arm, record in batches.items():
                if record['batch'].get('status') in {'completed', 'failed', 'expired', 'cancelled'}:
                    continue
                try:
                    record['batch'] = http('GET', ENDPOINT + '/' + record['batch']['id'])
                except (RuntimeError, urllib.error.URLError, TimeoutError):
                    continue
                save(state / (arm + '.json'), record)
                emit('N4_PROGRESS', {'arm': arm, **{k: record['batch'].get(k) for k in ['id', 'status', 'request_counts']}})
            time.sleep(30)
        save(out / 'batch-receipts.json', batches)
        events = []
        for arm, record in batches.items():
            batch = record['batch']
            assert batch.get('status') == 'completed', 'Terminal batch failure, no rescue'
            raw = batch.get('results') or []
            by_id = {r['custom_id']: r for r in raw}
            rr = [r for r in rows if r['arm'] == arm]
            assert len(raw) == len(by_id) and set(by_id) == {r['custom_id'] for r in rr}
            for row in rr:
                item = by_id[row['custom_id']]
                event = decode_response(row, item, {'allowed_served_models': protocol['models'][arm]['served']})
                if item.get('error') or (item.get('response') or {}).get('status_code') != 200:
                    event.update(valid=False, state='HTTP_OR_PROVIDER_ERROR')
                event.update(custom_id=row['custom_id'], arm=arm, doc_id=row['doc_id'], slot=row['slot'])
                events.append(event)
        save(out / 'outputs-before-gold.json', {'events': events, 'batches': batches})
        freeze = {'outputs_sha256': digest((out / 'outputs-before-gold.json').read_bytes()), 'unix': time.time(), 'gold_read': False}
        save(out / 'freeze.json', freeze)
        valid = all(e['valid'] for e in events)
        costs = {arm: record['batch'].get('usage') for arm, record in batches.items()}
        known_costs = [(value or {}).get('cost') for value in costs.values()]
        if all(type(value) in (int, float) for value in known_costs):
            assert all(math.isfinite(value) and value >= 0 for value in known_costs)
            assert sum(known_costs) <= ceiling + 1e-10, 'Reported provider cost exceeded reservation'
        if valid:
            labels = json.loads((frozen / 'evaluation-only.json').read_text())
            result = evaluate(public, events, labels)
        else:
            result = {'metrics': None, 'promotion': False, 'reason': 'Missing or invalid full-cohort responses, not negative votes'}
        result.update(status='COMPLETE_EXPLORATORY' if valid else 'BLOCKED_VALIDITY',
                      validity=dict(Counter(e['state'] for e in events)), costs=costs, freeze=freeze)
        save(out / 'results.json', result)
        export(out)
        emit('N4_RESULT', {k: v for k, v in result.items() if k != 'predictions'})


if __name__ == '__main__':
    main()
