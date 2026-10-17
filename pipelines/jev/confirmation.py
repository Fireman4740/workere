"""One-time confirmation: frozen configuration, label-free inference, no fitting."""
import hashlib
import json
import random
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from benchmark.configuration import read_yaml
from benchmark.io import read_json, write_json, write_jsonl
from benchmark.protocols import public_inputs, score_predictions, aggregate_statistics
from benchmark.schema import Prediction, Relation, probability
from benchmark.statistics import document_bootstrap, quantile
from benchmark.suite import documents
from pipelines.jev.client import DecisionClient
from pipelines.jev.state_builder import build_state


def project_scores(scores, *, symmetric):
    """Only probabilities enter this transformation; labels cannot affect it."""
    for (a, b), p in scores.items():
        if a == b or p is None:
            raise ValueError('invalid causal probability or self-pair')
        probability(p)
        if (b, a) not in scores:
            raise ValueError('both ordered candidates are required')
    return ({(a, b): (p + scores[b, a]) / 2 for (a, b), p in scores.items()}
            if symmetric else dict(scores))


def _load(path):
    path = Path(path).resolve()
    cfg = read_yaml(path)
    if cfg['schema'] != 'workere-esc-confirmation/v1':
        raise ValueError('invalid confirmation schema')
    inputs = {}
    for key in ('freeze', 'suite', 'selection'):
        source = path.parent / cfg[key]
        if hashlib.sha256(source.read_bytes()).hexdigest() != cfg[key + '_sha256']:
            raise ValueError(key + ' hash mismatch')
        inputs[key] = read_json(source)
    frozen, suite, selection = inputs['freeze'], inputs['suite'], inputs['selection']
    client = cfg['client']
    if (frozen['final_threshold'] != .65 or frozen['baseline_threshold'] != .5
            or frozen['final_mode'] != 'arithmetic_mean_of_two_orders'
            or frozen['predicted_coreference'] or frozen['oracle_coreference']
            or not frozen['no_refit_on_confirmation'] or frozen['CONF40_accessed_for_selection']):
        raise ValueError('confirmation configuration must retain the DEV freeze')
    if (client['model'] != frozen['decision_model']
            or client['model_version'] != frozen['decision_model_version']
            or client['expected_served_version'] != frozen['expected_served_model']
            or client['shared_budget']['limit_usd'] != frozen['global_budget_usd']
            or client['shared_budget']['scope'] != frozen['global_budget_scope']):
        raise ValueError('model or budget differs from the freeze')
    if (cfg['selection_sha256'] != frozen['selection_manifest_sha256']
            or selection['dataset_revision'] != frozen['dataset_revision']
            or suite['selection']['revision'] != frozen['dataset_revision']
            or suite['selection']['frozen_protocol_sha256'] != cfg['freeze_sha256']):
        raise ValueError('data revision or freeze provenance differs')
    docs = documents(suite)
    if ([d.id for d in docs] != frozen['confirmation_document_ids']
            or len(docs) != 40 or sum(len(d.pair_list) for d in docs) != 19174):
        raise ValueError('requires exactly the sealed CONF40 universe')
    client['cache_path'] = str(Path(client['cache_path']).expanduser())
    cfg['output_dir'] = str((path.parent / cfg['output_dir']).resolve())
    return cfg, frozen, suite


def plan(path):
    cfg, frozen, suite = _load(path)
    # Building model states uses public inputs only; no evaluation at planning time.
    overflows, calls = [], 0
    for item in public_inputs(suite, 'relations_gold'):
        refs = {e.id: e for e in item.reference_entities}
        for a, b in item.reference_candidates:
            state = build_state(item.document, refs[a], refs[b], frozen['context_style'])
            # Conservative UTF-8 byte bound exceeds token count for this text.
            if len(json.dumps({'state': state, 'questions': frozen['causal_questions']}, ensure_ascii=False).encode()) > cfg['client']['max_context_tokens']:
                overflows.append(item.document.id)
            calls += 1
    if overflows:
        raise ValueError('potential context overflow: '+str(sorted(set(overflows))))
    return {'schema': 'workere-esc-confirmation-plan/v1', 'name': cfg['name'],
            'documents': 40, 'calls_without_cache_or_retries': calls, 'api_calls_made': 0,
            'max_workers': cfg['max_workers'], 'threshold_frozen': .65,
            'no_refit_on_confirmation': True, 'freeze_sha256': cfg['freeze_sha256'],
            'global_budget_usd': 10, 'local_budget_usd': cfg['client']['budget_usd']}


def collect(cfg, frozen, suite, output):
    """Journal every attempt, stop queued tasks after failure, persist cache across resumes."""
    client = DecisionClient(**cfg['client'])
    lock, stop = threading.Lock(), threading.Event()
    stats = {'logical_queries': 0, 'api_calls': 0, 'cache_hits': 0,
             'known_api_cost_usd': 0.0, 'unknown_cost_events': 0}
    tasks = []
    for item in public_inputs(suite, 'relations_gold'):
        refs = {e.id: e for e in item.reference_entities}
        tasks.extend((item.document, refs[a], refs[b]) for a, b in item.reference_candidates)
    try:
        with (output / 'decisions.jsonl').open('x') as journal:
            def one(task):
                document, a, b = task
                if stop.is_set():
                    raise RuntimeError('another query failed; queued request stopped')
                state = build_state(document, a, b, frozen['context_style'])
                def record(event):
                    with lock:
                        journal.write(json.dumps({'document_id': document.id, 'source': a.id, 'target': b.id, **event})+'\n')
                        journal.flush()
                        stats['api_calls'] += bool(event.get('api_call'))
                        stats['cache_hits'] += bool(event['cache_hit'])
                        if event['cost_usd'] is None:
                            stats['unknown_cost_events'] += 1
                        else:
                            stats['known_api_cost_usd'] += event['cost_usd']
                try:
                    packet = client.decide(state, frozen['causal_questions'], on_attempt=record)
                    p = packet['response']['answers']['causal']['noul']
                    probability(p)
                    if p is None:
                        raise ValueError('missing causal probability')
                    with lock:
                        stats['logical_queries'] += 1
                        if stats['logical_queries'] % 100 == 0 or stats['logical_queries'] == len(tasks):
                            print('FINAL_PROGRESS '+json.dumps(stats), flush=True)
                    return {'id': document.id, 'source': a.id, 'target': b.id,
                            'positive_probability': p, 'cache_key': packet['cache_key']}
                except BaseException:
                    stop.set()
                    raise
            with ThreadPoolExecutor(max_workers=cfg['max_workers']) as pool:
                responses = list(pool.map(one, tasks))
        write_jsonl(output / 'responses.jsonl', responses)
        return responses, stats
    finally:
        stats['api_cost_usd'] = None if stats['unknown_cost_events'] else stats['known_api_cost_usd']
        write_json(output / 'resources.json', stats)
        client.close()


def predictions(suite, responses, *, symmetric, threshold):
    scores = {item.document.id: {} for item in public_inputs(suite, 'relations_gold')}
    for row in responses:
        key = (row['source'], row['target'])
        if key in scores[row['id']]:
            raise ValueError('duplicate causal response')
        scores[row['id']][key] = row['positive_probability']
    output = []
    for item in public_inputs(suite, 'relations_gold'):
        values = project_scores(scores[item.document.id], symmetric=symmetric)
        if set(values) != set(map(tuple, item.reference_candidates)):
            raise ValueError('incomplete candidate coverage')
        relations = tuple(Relation(a, b, 'causal' if p >= threshold else 'norel',
                                   p if p >= threshold else 1-p) for (a, b), p in values.items())
        metadata = {'method': 'frozen_mean_orders' if symmetric else 'original_ordered_baseline',
                    'threshold': threshold, 'direction_predicted': False,
                    'oracle_coreference': False, 'predicted_coreference': False,
                    'threshold_training_set': 'DEV-FULL', 'no_refit_on_confirmation': True,
                    'decision_type': 'noul', 'decision_scores': [
                        {'source': a, 'target': b, 'positive_probability': p} for (a, b), p in values.items()]}
        output.append(Prediction(item.document.id, item.reference_entities, relations, metadata).to_dict(item.document))
    return output


def paired_bootstrap(left, right, settings):
    a, b = left['document_statistics'], right['document_statistics']
    # The scorer emits records in suite order. Both runs use the identical suite.
    if [r['id'] for r in a] != [r['id'] for r in b]:
        raise ValueError('paired document IDs differ')
    rng, deltas = random.Random(settings['seed']), []
    for _ in range(settings['samples']):
        indices = [rng.randrange(len(a)) for _ in a]
        deltas.append(aggregate_statistics([b[i] for i in indices])['binary']['f1'] -
                      aggregate_statistics([a[i] for i in indices])['binary']['f1'])
    alpha = (1-settings['confidence_level'])/2
    return {'delta_f1': right['summary']['binary']['f1']-left['summary']['binary']['f1'],
            'confidence_interval': [quantile(deltas, alpha), quantile(deltas, 1-alpha)],
            'unit': 'document', **settings, 'threshold_refitted': False}


def run(path):
    cfg, frozen, suite = _load(path)
    output = Path(cfg['output_dir']) / (cfg['name']+'-'+uuid.uuid4().hex[:12])
    output.mkdir(parents=True)
    for name, data in [('suite', suite), ('parameters', cfg), ('final-freeze', frozen)]:
        write_json(output / (name+'.json'), data)
    print('FINAL_CONFIG '+json.dumps({'name': cfg['name'], 'run_dir': str(output), 'workers': cfg['max_workers'],
          'freeze_sha256': cfg['freeze_sha256'], 'suite_sha256': cfg['suite_sha256'],
          'baseline_threshold': frozen['baseline_threshold'], 'final_threshold': frozen['final_threshold'],
          'client': cfg['client']}), flush=True)
    responses, resources = collect(cfg, frozen, suite, output)
    # Persist BOTH prediction sets before first access to evaluation labels.
    rows = {name: predictions(suite, responses, symmetric=symmetric, threshold=threshold)
            for name, symmetric, threshold in [('baseline', False, frozen['baseline_threshold']),
                                               ('final', True, frozen['final_threshold'])]}
    for name, predictions_ in rows.items():
        write_jsonl(output / (name+'-predictions.jsonl'), predictions_)
    metrics = {}
    for name, predictions_ in rows.items():
        score = score_predictions(suite, predictions_, 'relations_gold', model=name)
        score['uncertainty'] = document_bootstrap(score['document_statistics'], **{
            'samples': frozen['paired_document_bootstrap']['samples'],
            'confidence': frozen['paired_document_bootstrap']['confidence_level'],
            'seed': frozen['paired_document_bootstrap']['seed']})
        write_json(output / (name+'-metrics.json'), score)
        metrics[name] = score
    comparison = paired_bootstrap(metrics['baseline'], metrics['final'], frozen['paired_document_bootstrap'])
    result = {'schema': 'workere-esc-confirmation-result/v1', 'status': 'completed', 'run_dir': str(output),
              'documents': 40, 'candidates': len(responses), 'freeze_sha256': cfg['freeze_sha256'],
              'variants': {name: m['summary'] for name, m in metrics.items()}, 'paired_final_minus_baseline': comparison,
              'resources': resources, 'no_refit_on_confirmation': True,
              'limitations': ['Binary causality; no direction prediction.', 'Luna comparison is between systems, not an isolated model ablation.']}
    write_json(output / 'result.json', result)
    return result
