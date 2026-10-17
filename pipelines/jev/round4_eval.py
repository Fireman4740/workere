"""Fixed-rule paired evaluation of the round-4 developmental screen; zero API calls."""
import itertools
import json
import random
from collections import defaultdict
from pathlib import Path

from benchmark.voting import aggregate


def stats(pred, gold, keys):
    tp = sum(pred[k] and gold[k] for k in keys)
    fp = sum(pred[k] and not gold[k] for k in keys)
    fn = sum(not pred[k] and gold[k] for k in keys)
    return tp, fp, fn


def f1(c):
    tp, fp, fn = c
    return 200 * tp / (2 * tp + fp + fn) if tp else 0.


def summary(pred, gold):
    tp, fp, fn = stats(pred, gold, gold)
    return {'P': round(100 * tp / max(tp + fp, 1), 3), 'R': round(100 * tp / max(tp + fn, 1), 3),
            'F1': round(f1((tp, fp, fn)), 3), 'tp': tp, 'fp': fp, 'fn': fn}


def paired(a, b, gold, unit):
    buckets = defaultdict(list)
    for k in gold:
        buckets[k[0] if unit == 'document' else k[0].split('_')[0]].append(k)
    units = sorted(buckets)
    per = {u: (stats(a, gold, buckets[u]), stats(b, gold, buckets[u])) for u in units}
    rng, values = random.Random(42), []
    for _ in range(2000):
        draw = [rng.choice(units) for _ in units]
        values.append(f1([sum(per[u][0][i] for u in draw) for i in range(3)]) -
                      f1([sum(per[u][1][i] for u in draw) for i in range(3)]))
    values.sort()
    return {'delta': round(f1(stats(a, gold, gold)) - f1(stats(b, gold, gold)), 3),
            'ci95': [round(values[50], 3), round(values[1949], 3)],
            'ci97_5_two_primary_contrasts': [round(values[25], 3), round(values[1974], 3)],
            'bootstrap_units': len(units), 'unit': unit}


def evaluate(receipt, control_path, variant, threshold=.39):
    if threshold != .39:
        raise ValueError('round-4 fusion threshold is frozen at .39')
    controls = json.loads(Path(control_path).read_text())
    gold, d1, old, new = {}, {}, {}, {}
    pass_triples = {}
    for r in controls:
        k = r['doc'], r['a'], r['b']
        if k in gold:
            raise ValueError('duplicate control candidate')
        gold[k], d1[k], old[k] = r['gold'], r['d1'], r['luna']
    for run in receipt['runs']:
        if run['status'] != 'completed':
            raise ValueError('incomplete round-4 screen; refusing a complete metric')
        for row in map(json.loads, (Path(run['run_dir']) / 'predictions.jsonl').read_text().splitlines()):
            doc, meta = row['id'], row['metadata']
            if doc in pass_triples:
                raise ValueError('duplicate evaluated document')
            pass_triples[doc] = meta['pass_triples']
            for s in meta['decision_scores']:
                k = doc, s['source'], s['target']
                if k in new:
                    raise ValueError('duplicate evaluated candidate')
                new[k] = s['positive_probability']
    if set(new) != set(gold) or not all((k[0], k[2], k[1]) in gold for k in gold):
        raise ValueError('new/control candidate coverage mismatch')
    def fused(scores):
        return {k: (scores[k] + scores[k[0], k[2], k[1]] + d1[k] + d1[k[0], k[2], k[1]]) / 4 >= threshold for k in gold}
    baseline, current = fused(old), fused(new)
    lu0 = {k: old[k] >= 4 / 7 - 1e-9 for k in gold}
    lu1 = {k: new[k] >= 4 / 7 - 1e-9 for k in gold}
    document = paired(current, baseline, gold, 'document')
    report = {'variant': variant, 'role': 'developmental screen on consumed DEV44; not confirmation',
              'documents': len(pass_triples), 'topics': len({d.split('_')[0] for d in pass_triples}),
              'ordered_pairs': len(gold), 'threshold': threshold,
              'control_fusion': summary(baseline, gold), 'new_fusion': summary(current, gold),
              'primary_paired_document': document, 'primary_paired_topic': paired(current, baseline, gold, 'topic'),
              'control_luna_binary_majority': summary(lu0, gold), 'new_luna_binary_majority': summary(lu1, gold),
              'secondary_luna_delta': paired(lu1, lu0, gold, 'document'),
              'development_screen_pass': document['delta'] >= 1 and document['ci97_5_two_primary_contrasts'][0] > 0}
    # Exact all-subset pass ablation, without any new requests or threshold fitting.
    ablation = {}
    for n in [1, 3]:
        fs, ls = [], []
        for subset in itertools.combinations(range(7), n):
            scores = {}
            for doc, triples in pass_triples.items():
                if len(triples) != 7:
                    raise ValueError('pass ablation requires all seven pass records')
                _, vote = aggregate([triples[i] for i in subset], 'norel', random.Random(42))
                for k in (k for k in gold if k[0] == doc):
                    counts = vote.get(k[1] + ',' + k[2], {}).get('vote_counts', {})
                    scores[k] = sum(v for lab, v in counts.items() if lab != 'norel') / n
            fs.append(summary(fused(scores), gold)['F1'])
            ls.append(summary({k: scores[k] > .5 for k in gold}, gold)['F1'])
        ablation[str(n)] = {'subsets': len(fs), 'mean_fusion_F1': round(sum(fs) / len(fs), 3),
                            'fusion_F1_range': [min(fs), max(fs)], 'mean_luna_F1': round(sum(ls) / len(ls), 3)}
    report['secondary_pass_ablation_no_refit'] = ablation
    return report
