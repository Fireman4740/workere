"""CPU-only, post-hoc audit of frozen W1/W2 predictions. No model inference."""
from __future__ import annotations

import argparse
import csv
from itertools import combinations
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from steering import ctb
from steering.weight_space import counts, digest, final_pairs, metrics, paired_interval
from steering.wsp_pilot import read_rows, sha, unique_rows, write_json


def strict_vote(predictions):
    """A reader contributes at most one vote per ordered pair; ties are rejected."""
    if not predictions:
        raise ValueError('vote requires at least one reader')
    frequency = {}
    for prediction in predictions:
        for pair in set(prediction):
            frequency[pair] = frequency.get(pair, 0) + 1
    minimum = len(predictions) // 2 + 1
    return {pair for pair, n in frequency.items() if n >= minimum}


def load_source(path, spec, selected):
    path = Path(path)
    for name, expected in spec['file_hashes'].items():
        if sha(path / name) != expected:
            raise ValueError(f'cached source changed: {name}')
    manifest = json.loads((path / 'manifest.json').read_text())
    if (manifest['id'] != spec['manifest_id'] or
            manifest['id'] != digest({k: v for k, v in manifest.items() if k != 'id'})):
        raise ValueError('source manifest identity mismatch')
    if sha(ctb.DATA / 'ctb-suite.json') != manifest['source_sha256']['suite']:
        raise ValueError('source corpus changed')
    implementation = manifest.get('frozen_implementation_sha256', manifest.get('implementation_sha256'))
    for name, expected in implementation.items():
        if sha(HERE / name) != expected:
            raise ValueError(f'W1 inference/scoring helper changed: {name}')
    ids = manifest['documents'] if 'documents' in manifest else manifest['split']['screen']
    conditions = {c['name']: c for c in manifest['conditions']}
    if len(ids) != 20 or not set(selected) <= set(conditions):
        raise ValueError('unexpected document/condition coverage')
    rows = unique_rows(read_rows(path / spec['records']), lambda r: (r['condition'], r['doc_id']))
    if set(rows) != {(name, d) for name in conditions for d in ids}:
        raise ValueError('missing or foreign source records')
    suite = ctb.load_suite()
    gold = {d: {(a, b) for a, label, b in suite[d]['gold'] if label != 'norel'} for d in ids}
    scores = json.loads((path / spec['scores']).read_text())
    if not scores['complete'] or scores['manifest_id'] != manifest['id']:
        raise ValueError('source scores incomplete or mismatched')
    predictions, tokens, invalid = {}, {}, {}
    for name, arm in conditions.items():
        predictions[name], tokens[name], invalid[name], per_doc = {}, {}, 0, {}
        for d in ids:
            row = rows[(name, d)]
            if (row['manifest_id'] != manifest['id'] or
                    row['directions_sha256'] != spec['file_hashes']['directions.npz']):
                raise ValueError('source row provenance mismatch')
            if any(row[k] != arm[k] for k in ('alpha', 'budget', 'temperature')) or row['pass'] != 0:
                raise ValueError('source generation settings mismatch')
            if 'condition_sha256' in row and row['condition_sha256'] != digest(arm):
                raise ValueError('source condition fingerprint mismatch')
            parsed = final_pairs(row)
            invalid[name] += parsed is None
            predictions[name][d] = parsed or set()
            tokens[name][d] = row['num_tokens']
            per_doc[d] = counts(predictions[name][d], gold[d])
        if per_doc != scores['per_document'][name]:
            raise ValueError(f'raw-pair recount differs from source score: {name}')
        if any(scores['conditions'][name][k] != v for k, v in metrics(per_doc).items()):
            raise ValueError(f'aggregate recount differs from source score: {name}')
    return ids, gold, {n: predictions[n] for n in selected}, {n: tokens[n] for n in selected}, invalid


def audit_predictions(ids, gold, predictions, tokens, plan):
    if not ids or len(set(ids)) != len(ids) or set(gold) != set(ids):
        raise ValueError('gold must include each document exactly once')
    if set(predictions) != set(plan['conditions']) or set(tokens) != set(predictions):
        raise ValueError('reader coverage mismatch')
    for name in predictions:
        if set(predictions[name]) != set(ids) or set(tokens[name]) != set(ids):
            raise ValueError('all readers must retain every document')
    per_doc = {name: {d: counts(pred[d], gold[d]) for d in ids}
               for name, pred in predictions.items()}
    total_base_tokens = sum(tokens['base'].values())
    results = {}

    def summarize(name, members):
        cost = sum(tokens[n][d] for n in members for d in ids)
        return {**metrics(per_doc[name]), 'documents': len(ids), 'readings_per_document': len(members),
                'emitted_tokens_total': cost, 'mean_emitted_tokens': cost / len(ids),
                'emitted_tokens_ratio_to_single_base': cost / total_base_tokens if total_base_tokens else None}

    for name in predictions:
        results[name] = summarize(name, [name])
    for vote in plan['votes']:
        name, members = vote['name'], vote['members']
        if (not members or len(members) != len(set(members)) or
                not set(members) <= set(predictions) or vote['minimum_votes'] != len(members) // 2 + 1):
            raise ValueError('only fixed strict-majority votes over distinct readers are permitted')
        per_doc[name] = {d: counts(strict_vote([predictions[n][d] for n in members]), gold[d]) for d in ids}
        results[name] = {**summarize(name, members),
                         'vs_single_base_f1_pp_exploratory': paired_interval(per_doc['base'], per_doc[name])}

    names = plan['conditions']
    unions = {d: set().union(*(predictions[n][d] for n in names)) for d in ids}
    union_counts = {d: counts(unions[d], gold[d]) for d in ids}
    missed = {d: sorted(gold[d] - unions[d]) for d in ids if gold[d] - unions[d]}
    contributions = {}
    for n in names:
        other_names = [m for m in names if m != n]
        contributions[n] = {
            'gold_recovered_vs_base_ordered': sum(len((predictions[n][d] - predictions['base'][d]) & gold[d]) for d in ids),
            'gold_lost_vs_base_ordered': sum(len((predictions['base'][d] - predictions[n][d]) & gold[d]) for d in ids),
            'gold_unique_to_this_reader_ordered': sum(len((predictions[n][d] -
                 set().union(*(predictions[m][d] for m in other_names))) & gold[d]) for d in ids)}
    pairwise = []
    for a, b in combinations(names, 2):
        intersection = sum(len(predictions[a][d] & predictions[b][d]) for d in ids)
        union = sum(len(predictions[a][d] | predictions[b][d]) for d in ids)
        pairwise.append({'a': a, 'b': b, 'edge_jaccard': intersection / union if union else None,
                         'both_empty_documents': sum(not predictions[a][d] and not predictions[b][d] for d in ids),
                         'gold_disagreements_ordered': sum(len((predictions[a][d] ^ predictions[b][d]) & gold[d]) for d in ids),
                         'shared_false_positives_ordered': sum(len((predictions[a][d] & predictions[b][d]) - gold[d]) for d in ids)})
    return {'conditions': results, 'per_document': per_doc,
            'union_diagnostic': {**metrics(union_counts), 'gold_positive_documents': sum(bool(gold[d]) for d in ids),
                 'gold_recovered_beyond_base_ordered': sum(len((unions[d] - predictions['base'][d]) & gold[d]) for d in ids),
                 'missed_by_every_reader': missed,
                 'interpretation': 'Union recall is the ceiling for any selector restricted to these proposals; precision requires a selector.'},
            'reader_contributions': contributions, 'pairwise': pairwise,
            'scope': plan['scope'], 'missing_control': plan['missing_control'], 'cost_caveat': plan['cost']}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-screen', required=True)
    p.add_argument('--source-confirmation', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--plan', default=str(HERE / 'wsp_diversity_plan.json'))
    args = p.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {'plan': plan, 'code_sha256': sha(__file__), 'plan_sha256': sha(args.plan)}
    manifest['id'] = digest(manifest)
    path = output / 'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('audit manifest drift')
    write_json(path, manifest)
    sources = {'W1': args.source_screen, 'W2': args.source_confirmation}
    report, documents, csv_rows = {}, {}, []
    for cohort, source in sources.items():
        ids, gold, predictions, tokens, invalid = load_source(source, plan['sources'][cohort], plan['conditions'])
        documents[cohort] = set(ids)
        result = audit_predictions(ids, gold, predictions, tokens, plan)
        result.update(source_run=plan['sources'][cohort]['run_id'], source_invalid_answers=invalid,
                      all_source_records_recounted=True)
        report[cohort] = result
        for name, values in result['conditions'].items():
            csv_rows.append({'cohort': cohort, 'condition': name, **{k: v for k, v in values.items() if not isinstance(v, list)}})
        print('WSP_DIVERSITY_RESULT ' + json.dumps({'cohort': cohort, **result}), flush=True)
    if documents['W1'] & documents['W2']:
        raise ValueError('source cohorts overlap')
    write_json(output / 'diversity_audit.json', {'manifest_id': manifest['id'], 'cohorts': report,
                                               'next': plan['next'], 'automatic_promotion': False})
    with (output / 'conditions.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print('WSP_DIVERSITY_COMPLETE ' + json.dumps({'manifest_id': manifest['id'],
          'cohorts': list(report), 'new_generations': 0, 'automatic_promotion': False, 'next': plan['next']}), flush=True)


if __name__ == '__main__':
    main()
