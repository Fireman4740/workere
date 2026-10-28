"""Independent stdlib replay: python audit_n10.py <N10 extracted evidence directory>."""
import collections
import hashlib
import itertools
import json
import math
import pathlib
import random
import sys

p = pathlib.Path(sys.argv[1])
load = lambda name: json.loads((p/name).read_text())
hashfile = lambda name: hashlib.sha256((p/name).read_bytes()).hexdigest()
manifest = load('manifest.json')
for name, expected in manifest['files_sha256'].items():
    assert hashfile(name) == expected, name
protocol = load('protocol.json')
bundle = load('inputs.json')
freeze = load('freeze-before-runtime-gold.json')
assert freeze['controls_parsed'] is False and freeze['historical_gold_exposure_acknowledged'] is True
assert freeze['predictions_sha256'] == hashfile('predictions-before-runtime-gold.json')
assert freeze['protocol_sha256'] == hashfile('protocol.json') and freeze['inputs_sha256'] == hashfile('inputs.json')
assert protocol['input_files_sha256']['inputs.json'] == hashfile('inputs.json')
assert protocol['input_files_sha256']['controls.json'] == hashfile('controls.json')
assert protocol['resources']['new_model_calls'] == protocol['resources']['new_API_usd'] == 0
public = {x['id']: {tuple(y) for y in x['pair_list']} for x in bundle['public']}
assert list(public) == protocol['document_ids'] and len(public) == 16
assert len({x.split('_')[0] for x in public}) == 16
scores = {(x['doc'], x['a'], x['b']): x['d1'] for x in bundle['d1']}
universe = {(doc, a, b) for doc, pairs in public.items() for a, b in pairs}
assert set(scores) == universe and len(scores) == len(bundle['d1'])
assert all(math.isfinite(v) and 0 <= v <= 1 for v in scores.values())
answers = {}
response_ids = set()
batch_counts = collections.Counter()
for e in bundle['events']:
    key = (e['arm'], e['doc'], e['schedule']*3+e['slot'])
    assert key not in answers and key[0] in ['ordinary', 'contrastive'] and 0 <= key[2] < 6
    assert e['logical_id'] == f"s{e['schedule']}:{e['arm']}:{e['doc']}:{e['slot']}"
    assert e['response_id'] and e['response_id'] not in response_ids
    response_ids.add(e['response_id'])
    assert e['served_model'] in {'openai/gpt-6-luna', 'openai/gpt-6-luna:batch', 'openai/gpt-6-luna-20260922'}
    assert e['provider'] == 'OpenAI' and e['finish_reason'] == 'stop' and e['error'] is None
    assert e['api_call_in_N7'] is True and e['cache_hit_in_N7'] is False
    assert e['request_model'] == 'openai/gpt-6-luna:batch' and e['reasoning'] == {'effort':'high'} and e['max_tokens'] == 32768
    b = bundle['batches'][e['batch_id']]
    assert b['model'] == 'openai/gpt-6-luna-20260922' and b['provider'] == 'OpenAI' and b['status'] == 'completed'
    assert b['request_counts'] == {'total':3,'completed':3,'failed':0}
    assert b['responses'][e['response_id']] == {'content':e['content'], 'response_sha256':e['response_sha256']}
    assert b['usage']['cost'] == e['provider_reported_batch_cost_usd']
    assert b['source_sha256'] == bundle['source_manifest']['files_sha256'][b['source_file']]
    batch_counts[e['batch_id']] += 1
    rows = json.loads(e['content'])
    assert isinstance(rows, list)
    answer = {}
    for row in rows:
        assert isinstance(row, dict) and set(row) == {'pair','label'}
        assert row['label'] in ['causes','causedby','norel']
        pair = tuple(x.strip() for x in row['pair'].split(','))
        assert pair in public[e['doc']]
        answer[pair] = int(row['label'] != 'norel')
    answers[key] = answer
expected_slots = {(a,d,i) for a in ['ordinary','contrastive'] for d in public for i in range(6)}
assert set(answers) == expected_slots and len(answers) == 192
assert len(bundle['batches']) == len(batch_counts) == 64 and set(batch_counts.values()) == {3}
assert len({e['request_body_sha256'] for e in bundle['events']}) == 135
subsets = [tuple(range(6))] + list(itertools.combinations(range(6), 3))
names = ['full6'] + ['three_'+''.join(str(i) for i in subset) for subset in subsets[1:]]
assert len(subsets) == 21
predictions = load('predictions-before-runtime-gold.json')
# Replay every prediction using only final answers, d1 and public candidate pairs before labels.
for arm in ['ordinary','contrastive']:
    for name, subset in zip(names, subsets):
        for doc, pairs in public.items():
            out = []
            for a,b in sorted(pairs):
                forward = sum(answers[arm,doc,i].get((a,b),0) for i in subset)/len(subset)
                reverse = sum(answers[arm,doc,i].get((b,a),0) for i in subset)/len(subset)
                if (forward+reverse+scores[doc,a,b]+scores[doc,b,a])/4 >= .39:
                    out.append([a,b])
            assert predictions[arm][name][doc] == out, (arm,name,doc)
controls = load('controls.json')
gold = {(r['doc'],r['a'],r['b']):r['gold'] for r in controls}
assert set(gold) == universe and len(gold) == len(controls)
assert all(type(v) is bool for v in gold.values())
assert all(gold[d,a,b] == gold[d,b,a] for d,a,b in gold)
counts = {}
for arm in ['ordinary','contrastive']:
    counts[arm] = {}
    for name in names:
        by_topic = collections.defaultdict(lambda:[0,0,0])
        proposed = {(d,a,b) for d, pairs in predictions[arm][name].items() for a,b in pairs}
        for key, truth in gold.items():
            pred = key in proposed
            row = by_topic[key[0].split('_')[0]]
            if pred and truth: row[0] += 1
            elif pred: row[1] += 1
            elif truth: row[2] += 1
        counts[arm][name] = dict(by_topic)
assert counts == load('per-topic-counts.json')
topics = sorted(counts['ordinary']['full6'])

def f1(c):
    return 200*c[0]/(2*c[0]+c[1]+c[2]) if c[0] else 0.0

def totals(arm, name, ts):
    return [sum(counts[arm][name][t][i] for t in ts) for i in range(3)]

def contrasts(ts):
    values = {arm:[f1(totals(arm,name,ts)) for name in names] for arm in ['ordinary','contrastive']}
    return [values['ordinary'][0]-sum(values['ordinary'][1:])/20,
            values['contrastive'][0]-sum(values['contrastive'][1:])/20,
            values['contrastive'][0]-values['ordinary'][0]]

estimates = contrasts(topics)
bs = protocol['bootstrap']
assert bs['draws'] == 20000 and bs['seed'] == 20261028010 and bs['family_hypotheses'] == 3
assert bs['lower_sorted_index'] == math.floor((.05/3/2)*bs['draws']) == 166
assert bs['upper_sorted_index'] == math.ceil((1-.05/3/2)*bs['draws'])-1 == 19833
rng = random.Random(bs['seed'])
bootstrap = [[],[],[]]
for _ in range(bs['draws']):
    values = contrasts(rng.choices(topics,k=len(topics)))
    for j, value in enumerate(values): bootstrap[j].append(value)
metrics = load('metrics.json')
for j, result in enumerate(metrics['primary_contrasts']):
    draws = sorted(bootstrap[j])
    ci = [draws[166],draws[19833]]
    assert result['hypothesis'] == protocol['primary_contrasts'][j]
    assert result['estimate_points'] == estimates[j] and result['Bonferroni_CI98_333333_points'] == ci
    assert result['exploratory_candidate'] == (estimates[j] >= 1 and ci[0] > 0)
all_subsets = load('all20-subset-statistics.json')
for arm in ['ordinary','contrastive']:
    for name in names:
        c = totals(arm,name,topics)
        assert all_subsets[arm][name] == {'TP_FP_FN':c,'F1_points':f1(c)}
    summary = metrics['systems'][arm]
    assert summary['full6'] == all_subsets[arm]['full6']
    assert summary['mean20three_subset_F1_points'] == sum(all_subsets[arm][n]['F1_points'] for n in names[1:])/20
    expected_counts = [sum(all_subsets[arm][n]['TP_FP_FN'][i] for n in names[1:])/20 for i in range(3)]
    assert summary['secondary_mean_three_counts'] == expected_counts
    assert summary['secondary_F1_of_mean_three_counts_points'] == f1(expected_counts)
historical_cost = sum(b['usage']['cost'] for b in bundle['batches'].values())
assert metrics['historical_N7_API_usd_already_accounted'] == historical_cost
assert metrics['new_API_usd'] == metrics['new_model_calls'] == 0 and metrics['production_promotion'] is False
print(json.dumps({'audit':'PASS','manifest_files_verified':len(manifest['files_sha256']),'cached_answers_replayed':192,
    'prediction_systems_recomputed':42,'topics':len(topics),'subsets_per_arm':20,'primary_contrasts':metrics['primary_contrasts'],
    'new_API_calls':0,'historical_N7_API_usd':historical_cost},indent=2))
