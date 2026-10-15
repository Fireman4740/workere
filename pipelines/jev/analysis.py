"""Offline probability diagnostics; every threshold/baseline uses the unchanged scorer."""
from __future__ import annotations

import copy
import csv
import random
from pathlib import Path

from benchmark.io import read_json, read_jsonl, write_json, write_jsonl
from benchmark.labels import BINARY_POS, NOREL, is_norel
from benchmark.metrics import prf
from benchmark.protocols import public_inputs, score_predictions
from benchmark.statistics import document_bootstrap, quantile
from benchmark.suite import documents
from pipelines.jev.postprocessor import noul_label


def _load(run_dir):
    directory = Path(run_dir)
    metadata = read_json(directory / 'metadata.json')
    if metadata['status'] != 'completed' or metadata['protocol'] != 'relations_gold':
        raise ValueError('analysis requires a completed reference-mention run without API errors')
    suite, rows = read_json(directory / 'suite.json'), read_jsonl(directory / 'predictions.jsonl')
    # Check complete coverage and oracle invariants using the official scoring adapter.
    score_predictions(suite, rows, 'relations_gold')
    for row, inputs in zip(rows, public_inputs(suite, 'relations_gold')):
        scores = row.get('metadata', {}).get('decision_scores', [])
        pairs = [(score['source'], score['target']) for score in scores]
        if row['id'] != inputs.document.id or len(pairs) != len(set(pairs)) or set(pairs) != set(inputs.reference_candidates):
            raise ValueError('recorded scores do not cover exactly the historical candidates')
        for score in scores:
            noul_label(score['positive_probability'])
    return directory, metadata, suite, rows


def threshold_predictions(rows, threshold):
    out = copy.deepcopy(rows)
    for row in out:
        scores = row['metadata']['decision_scores']
        row['relations'] = [{'source': score['source'], 'target': score['target'],
                            'label': noul_label(score['positive_probability'], threshold),
                            'confidence': score['positive_probability'] if score['positive_probability'] >= threshold
                            else 1 - score['positive_probability']} for score in scores]
        row['metadata'].update(decision_type='noul', threshold=threshold, direction_predicted=False,
                               offline_binary_projection=True)
    return out


def rethreshold(run_dir, threshold, output_dir):
    directory, metadata, suite, rows = _load(run_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    predictions = threshold_predictions(rows, threshold)
    metrics = score_predictions(suite, predictions, 'relations_gold', model=metadata['name'] + '-threshold')
    cfg = read_json(directory / 'parameters.json')['evaluation']
    metrics['uncertainty'] = document_bootstrap(metrics['document_statistics'], samples=cfg['bootstrap_samples'],
        confidence=cfg['confidence_level'], seed=cfg['bootstrap_seed'])
    write_jsonl(output / 'predictions.jsonl', predictions)
    write_json(output / 'metrics.json', metrics)
    write_json(output / 'provenance.json', {'source': str(directory.resolve()), 'threshold': threshold,
        'api_calls': 0, 'note': 'Binary causal existence only; no inferred direction.'})
    return metrics


def _distribution(values):
    return {'count': len(values), 'mean': sum(values) / len(values) if values else None,
            'quantiles': {str(q): quantile(values, q) if values else None for q in (0, .1, .25, .5, .75, .9, 1)}}


def _curve(samples):
    """Pair-level PR curve with tied probabilities grouped; ordered candidate universe."""
    ordered = sorted(samples, key=lambda r: -r['positive_probability'])
    support = sum(row['gold_positive'] for row in ordered)
    tp = fp = 0
    points = []
    for i, row in enumerate(ordered):
        tp += row['gold_positive']
        fp += not row['gold_positive']
        if i + 1 == len(ordered) or ordered[i + 1]['positive_probability'] != row['positive_probability']:
            p, r, f = prf(tp, fp, support - tp)
            points.append({'threshold': row['positive_probability'], 'precision': p, 'recall': r, 'f1': f,
                           'tp': tp, 'fp': fp, 'fn': support - tp})
    return points


def _order_consistency(samples):
    lookup = {(s['id'], s['source'], s['target']): s for s in samples}
    pairs = []
    for (doc, a, b), row in lookup.items():
        reverse = lookup.get((doc, b, a))
        if a >= b or reverse is None:
            continue
        pairs.append({'id': doc, 'pair': [a, b], 'absolute_probability_difference':
            abs(row['positive_probability'] - reverse['positive_probability']),
            'binary_disagreement': is_norel(row['predicted_label']) != is_norel(reverse['predicted_label']),
            'gold_binary_disagreement': row['gold_positive'] != reverse['gold_positive']})
    return {'paired_orders': len(pairs), 'binary_disagreements': sum(p['binary_disagreement'] for p in pairs),
        'gold_binary_disagreements': sum(p['gold_binary_disagreement'] for p in pairs),
        'mean_absolute_probability_difference': sum(p['absolute_probability_difference'] for p in pairs) / len(pairs) if pairs else None,
        'largest_differences': sorted(pairs, key=lambda p: -p['absolute_probability_difference'])[:10],
        'interpretation': 'Diagnostic of swapping A/B for causal existence, not a direction metric or proof of an error mechanism.'}


def analyze(run_dir, output_dir, *, random_seeds=(13, 42, 2026), random_positive_rate=.5):
    directory, metadata, suite, rows = _load(run_dir)
    noul_label(random_positive_rate)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    docs = {doc.id: doc for doc in documents(suite)}
    samples, issues, fp, fn = [], [], [], []
    official = score_predictions(suite, rows, 'relations_gold', model=metadata['name'])
    for row in rows:
        doc = docs[row['id']]
        spans = {mid: list(span) for mid, span in zip(doc.mentions, doc.spans)}
        gold = {(a, b): label for a, label, b in doc.gold}
        candidates = {tuple(pair) for pair in doc.pair_list}
        outside = [(a, label, b) for a, label, b in doc.gold if (a, b) not in candidates and not is_norel(label)]
        if outside:
            issues.append({'id': doc.id, 'gold_positives_outside_candidates': outside})
        predicted = {(r['source'], r['target']): r['label'] for r in row['relations']}
        confidences = {(r['source'], r['target']): r.get('confidence') for r in row['relations']}
        for score in row['metadata']['decision_scores']:
            pair = (score['source'], score['target'])
            gpos, ppos = not is_norel(gold.get(pair, NOREL)), not is_norel(predicted.get(pair, NOREL))
            sa, sb = doc.sentence_of.get(pair[0]), doc.sentence_of.get(pair[1])
            item = {**score, 'id': doc.id, 'gold_label': gold.get(pair, NOREL), 'gold_positive': gpos,
                    'predicted_label': predicted.get(pair, NOREL),
                    'predicted_confidence': confidences.get(pair),
                    'scope': 'unknown' if sa is None or sb is None else 'intra' if sa == sb else 'inter',
                    'source_surface': doc.surfaces.get(pair[0]), 'target_surface': doc.surfaces.get(pair[1]),
                    'source_token_indices': spans.get(pair[0]), 'target_token_indices': spans.get(pair[1]),
                    'selected_sentences': {str(s): ' '.join(doc.tokens[doc.sentences[s][0]:doc.sentences[s][1]])
                        for s in {sa, sb} if s is not None}}
            samples.append(item)
            if not gpos and ppos:
                fp.append(item)
            if gpos and not ppos:
                fn.append(item)
    sensitivity = []
    for threshold in [i / 20 for i in range(21)]:
        score = score_predictions(suite, threshold_predictions(rows, threshold), 'relations_gold')
        sensitivity.append({'threshold': threshold, **score['summary']})
    baseline_scores = []
    for seed in (None, *random_seeds):
        rng = random.Random(seed)
        name = 'always-negative' if seed is None else f'random-p{random_positive_rate}-seed{seed}'
        predictions = copy.deepcopy(rows)
        for row in predictions:
            row['relations'] = [{'source': s['source'], 'target': s['target'], 'label':
                BINARY_POS if seed is not None and rng.random() < random_positive_rate else NOREL,
                'confidence': None} for s in row['metadata']['decision_scores']]
            row['metadata'] = {'baseline': name, 'seed': seed,
                               'random_positive_rate': random_positive_rate if seed is not None else 0}
        score = score_predictions(suite, predictions, 'relations_gold', model=name)
        write_jsonl(output / (name + '-predictions.jsonl'), predictions)
        write_json(output / (name + '-metrics.json'), score)
        baseline_scores.append({'name': name, 'seed': seed, 'summary': score['summary'], 'api_calls': 0})
    report = {'schema': 'workere-jev-analysis/v1', 'source_run': str(directory.resolve()),
        'suite_fingerprint': suite['fingerprint'], 'api_calls': 0, 'documents': len(rows),
        'candidates': len(samples), 'candidate_errors': issues, 'summary': official['summary'],
        'directional_metrics_applicable': all(row['metadata'].get('direction_predicted', False) for row in rows),
        'probability_order_consistency': _order_consistency(samples),
        'confusion_matrix': {'tp': sum(s['gold_positive'] and not is_norel(s['predicted_label']) for s in samples),
            'fp': len(fp), 'fn': len(fn), 'tn': sum(not s['gold_positive'] and is_norel(s['predicted_label']) for s in samples)},
        'probability_distribution': {'gold_positive': _distribution([s['positive_probability'] for s in samples if s['gold_positive']]),
            'gold_negative': _distribution([s['positive_probability'] for s in samples if not s['gold_positive']])},
        'precision_recall_curve': _curve(samples), 'curve_universe': 'historical ordered candidate pairs; no synthetic probabilities for omitted gold',
        'threshold_sensitivity': sensitivity, 'baselines': baseline_scores,
        'threshold_selection': 'not performed; threshold remains 0.5',
        'threshold_sensitivity_interpretation': 'Binary projection from p(causal). Choice itself uses its structured chosen class, not a binary threshold.',
        'limitations': ['Noul predicts causal existence, not direction.',
            'A one-document bootstrap interval is degenerate and does not establish generalization.',
            'Historical baseline unavailable without the original report and exact matching suite.']}
    write_json(output / 'analysis.json', report)
    write_jsonl(output / 'false-positives.jsonl', sorted(fp, key=lambda r: -r['positive_probability']))
    write_jsonl(output / 'false-negatives.jsonl', sorted(fn, key=lambda r: r['positive_probability']))
    write_jsonl(output / 'pair-probabilities.jsonl', samples)
    fields = ['id', 'source', 'target', 'source_surface', 'target_surface', 'gold_label', 'predicted_label',
              'positive_probability', 'predicted_confidence', 'scope', 'served_model', 'cache_key']
    with (output / 'pair-probabilities.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(samples)
    return report
