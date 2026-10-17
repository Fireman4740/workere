"""Frozen ESC rules evaluated from recorded probabilities without inference."""
from __future__ import annotations
import hashlib
import json
import uuid
from collections import defaultdict
from pathlib import Path
from benchmark.configuration import read_yaml
from benchmark.io import read_json, write_json, write_jsonl
from benchmark.labels import is_norel
from benchmark.protocols import public_inputs, score_predictions
from benchmark.schema import Prediction, Relation, probability
from benchmark.statistics import document_bootstrap
from benchmark.suite import documents


def f1(count):
    tp, fp, fn = count
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


def counts(values, labels, threshold):
    return (sum(p >= threshold and g for p, g in zip(values, labels)),
            sum(p >= threshold and not g for p, g in zip(values, labels)),
            sum(p < threshold and g for p, g in zip(values, labels)))


def threshold_table(values, labels, grid):
    return {doc: {t: counts(values[doc], labels[doc], t) for t in grid} for doc in values}


def choose_threshold(table, training_ids, grid):
    training_ids = tuple(training_ids)
    if not training_ids:
        raise ValueError('threshold fitting requires training documents')
    def objective(t):
        pooled = tuple(sum(table[d][t][i] for d in training_ids) for i in range(3))
        return f1(pooled), t
    return max(grid, key=objective)


def lodo_thresholds(values, labels, grid):
    ids = sorted(values)
    if set(ids) != set(labels) or len(ids) < 2:
        raise ValueError('LODO requires at least two matching documents')
    table = threshold_table(values, labels, grid)
    return {held: choose_threshold(table, (d for d in ids if d != held), grid) for held in ids}


def symmetrize(scores):
    if any((b, a) not in scores for a, b in scores):
        raise ValueError('both candidate orders are required')
    return {(a, b): (p + scores[b, a]) / 2 for (a, b), p in scores.items()}


def aggregate_chains(scores, cluster):
    if any(a not in cluster or b not in cluster for a, b in scores):
        raise ValueError('coreference must cover all endpoints including singletons')
    maxima = defaultdict(float)
    for (a, b), p in scores.items():
        if cluster[a] != cluster[b]:
            key = frozenset((cluster[a], cluster[b]))
            maxima[key] = max(maxima[key], p)
    return {(a, b): maxima[frozenset((cluster[a], cluster[b]))] if cluster[a] != cluster[b] else 0.0
            for a, b in scores}


def _load(path):
    path = Path(path).resolve()
    cfg = read_yaml(path)
    if cfg.get('schema') != 'workere-esc-refinement/v1' or cfg.get('phase') != 'development':
        raise ValueError('this workflow only permits DEV-FULL fitting')
    if cfg.get('api_calls') != 0:
        raise ValueError('offline refinement cannot request model calls')
    input_path = path.parent / cfg['inputs']
    if hashlib.sha256(input_path.read_bytes()).hexdigest() != cfg['inputs_sha256']:
        raise ValueError('source probability input hash mismatch')
    data = read_json(input_path)
    if data.get('schema') != 'workere-esc-refinement-inputs/v1':
        raise ValueError('invalid refinement inputs')
    rules = read_yaml(path.parent / cfg['rules'])
    # YAML dates become datetime.date; retain their ISO representation in receipts.
    rules['registered'] = str(rules['registered'])
    if (rules['symmetrization'] != 'arithmetic_mean_of_ordered_probabilities'
            or rules['event_pair_aggregation'] != 'maximum_over_cross_chain_mention_pairs'
            or rules['within_chain'] != 'no_relation'
            or rules['threshold']['ties'] != 'largest_threshold'):
        raise ValueError('unsupported preregistration')
    docs = documents(data['suite'])
    selection = read_json(path.parent / cfg['selection'])
    expected = {d['id'] for d in selection['dev_full']}
    if {d.id for d in docs} != expected or set(data['scores']) != expected:
        raise ValueError('requires exactly DEV-FULL, never D15 or CONF40')
    if data['dataset_revision'] != selection['dataset_revision']:
        raise ValueError('dataset revision mismatch')
    values, labels, pairs, oracle = {}, {}, {}, {}
    for doc in docs:
        rows = data['scores'][doc.id]
        p = {(a, b): value for a, b, value in rows}
        if len(p) != len(rows) or set(p) != set(map(tuple, doc.pair_list)):
            raise ValueError('incomplete candidate scores')
        for value in p.values():
            probability(value)
            if value is None:
                raise ValueError('missing probability')
        pairs[doc.id] = [(a, b) for a, b, _ in rows]
        gold = {(a, b): not is_norel(lab) for a, lab, b in doc.gold}
        if any(positive and pair not in p for pair, positive in gold.items()):
            raise ValueError('positive gold outside the frozen candidate universe')
        labels[doc.id] = [gold.get(pair, False) for pair in pairs[doc.id]]
        values[doc.id] = p
        raw = data['oracle_coreference'][doc.id]
        if raw['mentions'] != doc.mentions:
            raise ValueError('oracle IDs differ from frozen mentions')
        cluster = {}
        for ci, indices in enumerate(raw['coref_clusters']):
            for i in indices:
                if type(i) is not int or not 0 <= i < len(doc.mentions) or doc.mentions[i] in cluster:
                    raise ValueError('invalid or overlapping oracle chains')
                cluster[doc.mentions[i]] = ('chain', ci)
        oracle[doc.id] = {m: cluster.get(m, ('singleton', m)) for m in doc.mentions}
    return cfg, data, rules, values, labels, pairs, oracle


def plan(path):
    cfg, data, rules, values, labels, pairs, oracle = _load(path)
    return {'schema': 'workere-esc-refinement-plan/v1', 'name': cfg['name'], 'api_calls_made': 0,
            'calls_without_cache_or_retries': 0, 'estimated_input_cost_usd_at_configured_rate': 0,
            'context_overflows': [], 'documents': len(values), 'ordered_candidates': sum(map(len, pairs.values())),
            'source_run': data['source_run'], 'source_predictions_sha256': data['source_predictions_sha256'],
            'threshold_grid': rules['threshold']['grid'], 'selection': 'document LODO; oracle excluded from mode selection'}


def run(path):
    cfg, data, rules, values, labels, pairs, oracle = _load(path)
    output = (Path(path).resolve().parent / cfg['output_dir']).resolve() / (cfg['name'] + '-' + uuid.uuid4().hex[:12])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'parameters.json', cfg)
    write_json(output / 'provenance.json', {k: data[k] for k in ('source_run', 'source_commit', 'source_predictions_sha256', 'dataset_revision')})
    write_json(output / 'suite.json', data['suite'])
    ids, grid = sorted(values), rules['threshold']['grid']
    symmetric = {d: symmetrize(values[d]) for d in ids}
    oracle_values = {d: aggregate_chains(symmetric[d], oracle[d]) for d in ids}
    arrays = {'baseline': {d: [values[d][pair] for pair in pairs[d]] for d in ids},
              'symmetric': {d: [symmetric[d][pair] for pair in pairs[d]] for d in ids},
              'oracle': {d: [oracle_values[d][pair] for pair in pairs[d]] for d in ids}}
    thresholds = {'baseline_t0.5': {d: .5 for d in ids}, 'symmetric_t0.5': {d: .5 for d in ids},
                  'symmetric_LODO': lodo_thresholds(arrays['symmetric'], labels, grid),
                  'oracle_t0.5': {d: .5 for d in ids}, 'oracle_LODO': lodo_thresholds(arrays['oracle'], labels, grid)}
    results = {}
    for name, thresholds_by_doc in thresholds.items():
        family = name.split('_')[0]
        predictions = []
        for item in public_inputs(data['suite'], 'relations_gold'):
            d, threshold = item.document.id, thresholds_by_doc[item.document.id]
            pp = arrays[family][d]
            relations = tuple(Relation(a, b, 'causal' if p >= threshold else 'norel', p if p >= threshold else 1-p)
                              for (a, b), p in zip(pairs[d], pp))
            metadata = {'method': name, 'threshold': threshold, 'direction_predicted': False,
                        'oracle_coreference': family == 'oracle', 'api_calls': 0,
                        'threshold_training_ids': [other for other in ids if other != d] if name.endswith('LODO') else [],
                        'decision_type': 'noul', 'decision_scores': [
                            {'source': a, 'target': b, 'positive_probability': p} for (a, b), p in zip(pairs[d], pp)]}
            predictions.append(Prediction(d, item.reference_entities, relations, metadata).to_dict(item.document))
        metrics = score_predictions(data['suite'], predictions, 'relations_gold', model=name)
        if name == 'baseline_t0.5' and metrics['summary'] != data['baseline_summary']:
            raise ValueError('source baseline does not reproduce official scores')
        metrics['uncertainty'] = document_bootstrap(metrics['document_statistics'], samples=2000, confidence=.95, seed=42)
        write_jsonl(output / (name + '-predictions.jsonl'), predictions)
        write_json(output / (name + '-metrics.json'), metrics)
        results[name] = {'summary': metrics['summary'], 'thresholds': thresholds_by_doc,
                         'oracle_coreference': family == 'oracle', 'uncertainty': metrics['uncertainty']}
        print('R2_METRIC ' + json.dumps({'variant': name, 'binary': metrics['summary']['binary']}), flush=True)
    # Fit all DEV only after selecting a deployable mode; oracle is ineligible.
    selected = ('symmetric_LODO' if results['symmetric_LODO']['summary']['binary']['f1'] >
                results['baseline_t0.5']['summary']['binary']['f1'] else 'baseline_t0.5')
    final_threshold = (choose_threshold(threshold_table(arrays['symmetric'], labels, grid), ids, grid)
                       if selected == 'symmetric_LODO' else .5)
    frozen = {'source_run': data['source_run'], 'selected_mode': selected, 'threshold': final_threshold,
              'fit_documents': ids, 'selection_set': 'DEV-FULL', 'CONF40_used': False,
              'oracle_eligible': False, 'api_calls': 0, 'rules': rules}
    write_json(output / 'frozen-candidate.json', frozen)
    report = {'schema': 'workere-esc-refinement-result/v1', 'status': 'completed', 'run_dir': str(output),
              'api_calls': 0, 'documents': len(ids), 'variants': results, 'frozen_candidate': frozen,
              'limitations': ['Development cross-validation only; CONF40 untouched.',
                              'Bootstrap conditions on held-out predictions; no threshold refitting per resample.',
                              'Oracle chains are diagnostic bounds, excluded from deployment selection.']}
    write_json(output / 'result.json', report)
    return report
