"""N10 cached, exhaustive six-versus-three analysis. Stdlib only; no model client."""
import base64
import collections
import datetime
import hashlib
import io
import itertools
import json
import math
import os
import pathlib
import random
import zipfile

P = pathlib.Path('experiments/jev/n10')
ARMS = ('ordinary', 'contrastive')
SUBSETS = tuple(itertools.combinations(range(6), 3))
SYSTEMS = ('full6',) + tuple('three_' + ''.join(map(str, s)) for s in SUBSETS)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(pathlib.Path(path).read_text())


def save(path, value):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(path).write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')


def f1(counts):
    tp, fp, fn = counts
    return 200 * tp / (2 * tp + fp + fn) if tp else 0.0


def average_subset_f1(counts):
    # Averaging the counts first changes the estimand and is NOT the primary.
    return sum(f1(c) for c in counts) / len(counts)


def parse_content(content, allowed):
    items = json.loads(content)
    assert isinstance(items, list), 'answer must be JSON array'
    labels = {}
    duplicates = 0
    for item in items:
        assert isinstance(item, dict) and set(item) == {'pair', 'label'}, 'answer schema'
        assert isinstance(item['pair'], str) and item['label'] in {'causes', 'causedby', 'norel'}, 'pair/label type'
        pair = tuple(x.strip() for x in item['pair'].split(','))
        assert pair in allowed, 'unknown or out-of-universe pair'
        duplicates += int(pair in labels)
        labels[pair] = item['label']
    return {pair for pair, label in labels.items() if label != 'norel'}, duplicates


def validate_inputs(bundle, protocol):
    public = {d['id']: {tuple(x) for x in d['pair_list']} for d in bundle['public']}
    assert list(public) == protocol['document_ids'] and len(public) == 16
    assert len({x.split('_')[0] for x in public}) == 16
    d1 = {(r['doc'], r['a'], r['b']): r['d1'] for r in bundle['d1']}
    expected_pairs = {(doc,) + pair for doc, pairs in public.items() for pair in pairs}
    assert set(d1) == expected_pairs and len(d1) == len(bundle['d1'])
    assert all(math.isfinite(v) and 0 <= v <= 1 for v in d1.values())
    assert all((d, b, a) in d1 for d, a, b in d1)
    expected_slots = {(arm, doc, schedule, slot) for arm in ARMS for doc in public for schedule in (0, 1) for slot in range(3)}
    votes, ids, logical_ids, group_batches, duplicate_rows = {}, set(), set(), collections.defaultdict(set), 0
    batch_counts = collections.Counter()
    for e in bundle['events']:
        key = (e['arm'], e['doc'], e['schedule'], e['slot'])
        assert key in expected_slots and key not in votes
        assert e['logical_id'] == f"s{e['schedule']}:{e['arm']}:{e['doc']}:{e['slot']}"
        assert e['logical_id'] not in logical_ids
        logical_ids.add(e['logical_id'])
        assert e['response_id'] and e['response_id'] not in ids
        ids.add(e['response_id'])
        assert e['served_model'] in {'openai/gpt-6-luna', 'openai/gpt-6-luna:batch', 'openai/gpt-6-luna-20260922'}
        assert e['provider'] == 'OpenAI' and e['finish_reason'] == 'stop' and e['error'] is None
        assert e['api_call_in_N7'] is True and e['cache_hit_in_N7'] is False
        assert e['request_model'] == 'openai/gpt-6-luna:batch'
        assert e['max_tokens'] == 32768 and e['reasoning'] == {'effort': 'high'}
        assert all(len(e[k]) == 64 for k in ['source_line_sha256', 'response_sha256', 'request_body_sha256'])
        batch = bundle['batches'][e['batch_id']]
        assert batch['model'] == 'openai/gpt-6-luna-20260922' and batch['provider'] == 'OpenAI'
        assert batch['status'] == 'completed' and batch['request_counts'] == {'total': 3, 'completed': 3, 'failed': 0}
        assert len(batch['responses']) == 3
        response = batch['responses'][e['response_id']]
        assert response['content'] == e['content'] and response['response_sha256'] == e['response_sha256']
        assert batch['usage']['cost'] == e['provider_reported_batch_cost_usd']
        source_hash = bundle['source_manifest']['files_sha256'][batch['source_file']]
        assert batch['source_sha256'] == source_hash
        group_batches[e['arm'], e['doc'], e['schedule']].add(e['batch_id'])
        batch_counts[e['batch_id']] += 1
        positive, duplicates = parse_content(e['content'], public[e['doc']])
        votes[key] = positive
        duplicate_rows += duplicates
    assert set(votes) == expected_slots and len(votes) == 192
    assert len(group_batches) == 64 and all(len(v) == 1 for v in group_batches.values())
    assert len(bundle['batches']) == len(batch_counts) == 64 and set(batch_counts.values()) == {3}
    assert len({next(iter(v)) for v in group_batches.values()}) == 64
    assert len({e['request_body_sha256'] for e in bundle['events']}) == 135
    return public, d1, votes, {'status': 'PASS', 'answers': len(votes), 'unique_response_ids': len(ids), 'distinct_completed_batches': len(batch_counts), 'duplicate_rows_last_label_wins': duplicate_rows, 'new_model_calls': 0}


def predict(public, d1, votes):
    predictions = {}
    for arm in ARMS:
        predictions[arm] = {}
        for name, indices in zip(SYSTEMS, (tuple(range(6)),) + SUBSETS):
            docs = {}
            for doc, pairs in public.items():
                positive = []
                for a, b in sorted(pairs):
                    scores = []
                    for x, y in [(a, b), (b, a)]:
                        scores.append(sum((x, y) in votes[arm, doc, i // 3, i % 3] for i in indices) / len(indices))
                    fused = (scores[0] + scores[1] + d1[doc, a, b] + d1[doc, b, a]) / 4
                    if fused >= .39:
                        positive.append([a, b])
                docs[doc] = positive
            predictions[arm][name] = docs
    return predictions


def make_counts(predictions, controls):
    gold = {(r['doc'], r['a'], r['b']): r['gold'] for r in controls}
    assert len(gold) == len(controls) and all(type(v) is bool for v in gold.values())
    assert all(gold[d, a, b] == gold[d, b, a] for d, a, b in gold)
    per_topic = {}
    for arm in ARMS:
        per_topic[arm] = {}
        for name in SYSTEMS:
            predicted = {(doc, a, b) for doc, pairs in predictions[arm][name].items() for a, b in pairs}
            assert predicted <= set(gold)
            counts = collections.defaultdict(lambda: [0, 0, 0])
            for k, y in gold.items():
                pred = k in predicted
                c = counts[k[0].split('_')[0]]
                c[0] += int(pred and y)
                c[1] += int(pred and not y)
                c[2] += int(not pred and y)
            per_topic[arm][name] = dict(counts)
    return per_topic


def evaluate(per_topic, protocol):
    topics = sorted(per_topic['ordinary']['full6'])
    systems = [(a, s) for a in ARMS for s in SYSTEMS]
    arrays = [[per_topic[a][s][t] for t in topics] for a, s in systems]

    def sample_scores(multiplicities):
        values = []
        for rows in arrays:
            tp = fp = fn = 0
            for i, n in enumerate(multiplicities):
                if n:
                    c = rows[i]
                    tp += n * c[0]
                    fp += n * c[1]
                    fn += n * c[2]
            values.append(f1((tp, fp, fn)))
        return values

    def contrasts(values):
        return [values[0] - sum(values[1:21]) / 20,
                values[21] - sum(values[22:42]) / 20,
                values[21] - values[0]]

    values = sample_scores([1] * len(topics))
    estimates = contrasts(values)
    boot = [[], [], []]
    rng = random.Random(protocol['bootstrap']['seed'])
    for _ in range(protocol['bootstrap']['draws']):
        draw = rng.choices(topics, k=len(topics))
        freq = collections.Counter(draw)
        ds = contrasts(sample_scores([freq[t] for t in topics]))
        for j, value in enumerate(ds):
            boot[j].append(value)
    intervals = []
    for draws in boot:
        draws.sort()
        intervals.append([draws[protocol['bootstrap']['lower_sorted_index']], draws[protocol['bootstrap']['upper_sorted_index']]])
    summary = {}
    all_subsets = {}
    for ai, arm in enumerate(ARMS):
        names = {}
        for name in SYSTEMS:
            counts = [sum(per_topic[arm][name][t][i] for t in topics) for i in range(3)]
            names[name] = {'TP_FP_FN': counts, 'F1_points': f1(counts)}
        mean_counts = [sum(names[s]['TP_FP_FN'][i] for s in SYSTEMS[1:]) / 20 for i in range(3)]
        summary[arm] = {'full6': names['full6'], 'mean20three_subset_F1_points': sum(names[s]['F1_points'] for s in SYSTEMS[1:]) / 20,
                        'secondary_mean_three_counts': mean_counts, 'secondary_F1_of_mean_three_counts_points': f1(mean_counts),
                        'subset_F1_range_points': [min(names[s]['F1_points'] for s in SYSTEMS[1:]), max(names[s]['F1_points'] for s in SYSTEMS[1:])]}
        all_subsets[arm] = names
    results = []
    for name, estimate, ci in zip(protocol['primary_contrasts'], estimates, intervals):
        results.append({'hypothesis': name, 'estimate_points': estimate, 'Bonferroni_CI98_333333_points': ci,
                        'exploratory_candidate': estimate >= 1 and ci[0] > 0})
    return {'study': 'N10', 'status': 'ANSWERED_EXPLORATORY_EXPOSED_CACHE', 'topics': topics, 'systems': summary,
            'primary_contrasts': results, 'new_model_calls': 0, 'new_API_usd': 0,
            'production_promotion': False, 'limitations': ['Cached source outputs and labels already exposed; new analysis is exploratory.',
            'Bootstrap conditional on six available generations; does not estimate between-generation, model-version, or provider variability.',
            'N7 ordinary demos differ across schedules; contrastive demos repeat. Six-read effect includes that composition.',
            'Comparator is the mean over fixed cached three-of-six subsets, including mixed schedules/repeated demos, not a fresh three-read policy.',
            'Six reads need twice the generation count of three in production; zero new API cost here is reuse only.']}, all_subsets


def export(out):
    files = {str(p.relative_to(out)): sha(p.read_bytes()) for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'manifest.json'}
    save(out / 'manifest.json', {'run': os.environ.get('ORX_RUN_ID'), 'commit': os.environ.get('ORX_COMMIT'), 'files_sha256': files})
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, 'w', zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(out.rglob('*')):
            if p.is_file():
                archive.write(p, str(p.relative_to(out)))
    data = bio.getvalue()
    encoded = base64.b64encode(data).decode()
    chunks = [encoded[i:i+60000] for i in range(0, len(encoded), 60000)]
    for i, chunk in enumerate(chunks):
        print('N10_EXPORT_CHUNK ' + json.dumps({'index': i, 'data': chunk}), flush=True)
    print('N10_EXPORT_END ' + json.dumps({'chunks': len(chunks), 'bytes': len(data), 'sha256': sha(data)}), flush=True)


def main():
    assert os.environ.get('ORX_RUN_ID'), 'Scientific computation must run under tracked execution'
    protocol = read(P / 'protocol.json')
    # Checksums touch label-file bytes, but no label is parsed before predictions are frozen.
    for name, expected in protocol['input_files_sha256'].items():
        assert sha((P / name).read_bytes()) == expected, name
    bundle = read(P / 'inputs.json')
    public, d1, votes, validity = validate_inputs(bundle, protocol)
    predictions = predict(public, d1, votes)
    out = pathlib.Path('logs/n10-stabilite')
    save(out / 'predictions-before-runtime-gold.json', predictions)
    save(out / 'protocol.json', protocol)
    save(out / 'validity.json', validity)
    freeze = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'controls_parsed': False,
              'historical_gold_exposure_acknowledged': True, 'predictions_sha256': sha((out / 'predictions-before-runtime-gold.json').read_bytes()),
              'protocol_sha256': sha((P / 'protocol.json').read_bytes()), 'inputs_sha256': sha((P / 'inputs.json').read_bytes())}
    save(out / 'freeze-before-runtime-gold.json', freeze)
    print('N10_PREDICTIONS_FROZEN ' + json.dumps(freeze), flush=True)
    controls = read(P / 'controls.json')
    expected_pairs = {(d,) + pair for d, pairs in public.items() for pair in pairs}
    assert {(r['doc'], r['a'], r['b']) for r in controls} == expected_pairs
    counts = make_counts(predictions, controls)
    metrics, subsets = evaluate(counts, protocol)
    historical_cost = sum(b['usage']['cost'] for b in bundle['batches'].values())
    metrics['historical_N7_API_usd_already_accounted'] = historical_cost
    save(out / 'per-topic-counts.json', counts)
    save(out / 'all20-subset-statistics.json', subsets)
    save(out / 'metrics.json', metrics)
    save(out / 'costs.json', {'new_API_usd': 0, 'new_model_calls': 0, 'historical_N7_provider_reported_usd': historical_cost,
                            'historical_cost_counted_in_prior_campaign_only': True, 'production_six_vs_three_generation_ratio': 2,
                            'server_resources_not_priced': True})
    for name in ['inputs.json', 'controls.json']:
        (out / name).write_bytes((P / name).read_bytes())
    for source, target in [(pathlib.Path(__file__), 'controller.py'), (P / 'audit_n10.py', 'audit_n10.py'), (P / 'test_n10.py', 'test_n10.py')]:
        (out / target).write_bytes(source.read_bytes())
    save(out / 'execution.json', {'run': os.environ.get('ORX_RUN_ID'), 'commit': os.environ.get('ORX_COMMIT'),
                                'controls_opened_after_freeze_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                'new_model_calls': 0, 'source_run': bundle['source_run']})
    print('N10_RESULT ' + json.dumps(metrics), flush=True)
    export(out)


if __name__ == '__main__':
    main()
