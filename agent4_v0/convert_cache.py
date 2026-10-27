"""Allowlist-only boundary: historical DEV44 caches -> no-gold baseline and hash cohort.
Never import the historical Scorer or read an evaluated gold value.
"""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from .kit import pilot

from .common import jsonl, literal, load, probability, require, save, sha, token_offsets, validate_doc


def ids(items):
    return {x['id'] if isinstance(x, dict) else x for x in items}


def convert(reference_root, reserve_path, out, seed='agent4-v0-20261027'):
    root, out = Path(reference_root), Path(out)
    require(not out.exists() or not any(out.iterdir()), 'conversion requires fresh output directory')
    files = {
        'controls': root / 'experiments/jev/round4/controls.json',
        'development_selection': root / 'experiments/jev/round4/selection.json',
        'initial_selection': root / 'experiments/jev/document-selection.json',
        'round3_selection': root / 'experiments/jev/round3/round3-selection.json',
        'reserve_register': Path(reserve_path),
    }
    selection = load(files['development_selection'])
    initial, round3, reserve = (load(files[k]) for k in ('initial_selection', 'round3_selection', 'reserve_register'))
    require(selection['selection_uses_labels'] is False, 'development selection not label-free')
    excluded = ids(initial['confirmation']['docs']) | ids(round3['test_s40']) | ids(round3['holdout_rest'])
    excluded |= ids(reserve['remaining_unused_reserve']) | ids(reserve['reserved_for_j3_confirmation'])
    allowed = ids(selection['document_ids']) - excluded
    # Mixed source contains a gold column. Copy only these five public fields; never branch on gold.
    raw = load(files['controls'])
    signals = defaultdict(dict)
    for row in raw:
        did = row['doc']
        if did not in allowed:
            continue
        a, b = str(row['a']), str(row['b'])
        require((a, b) not in signals[did], 'duplicate ordered cached pair')
        signals[did][a, b] = [probability(row['luna']), probability(row['d1'])]
    baseline, rejected = [], []
    for did in sorted(allowed):
        source = root / 'experiments/jev/round4/suites' / f'{did}.json'
        files['suite:' + did] = source
        s = load(source)['documents'][0]
        # Explicit field allowlist, not a shallow copy which could leak gold or chains.
        doc = {k: literal(s[k]) for k in ('tokens', 'mentions', 'spans', 'sentences')}
        doc['mentions'] = [str(m) for m in doc['mentions']]
        doc.update(doc_id=did, topic=did.split('_')[0], text=s['text'],
                   candidates=[[str(a), str(b)] for a, b in literal(s['pair_list'])],
                   offsets=token_offsets(doc['tokens']))
        validate_doc(doc)
        require(set(signals[did]) == {tuple(p) for p in doc['candidates']}, 'cache candidate coverage differs')
        pairs = []
        for a, b in sorted(signals[did]):
            if a >= b:
                continue
            require((b, a) in signals[did], 'missing reverse score; never copy forward silently')
            v_ab, d_ab = signals[did][a, b]
            v_ba, d_ba = signals[did][b, a]
            pairs.append({'pair': [a, b], 'vote_ab': v_ab, 'vote_ba': v_ba,
                          'd1_ab': d_ab, 'd1_ba': d_ba, 'b': (v_ab + v_ba + d_ab + d_ba) / 4})
        # The supplied kit represents contiguous, non-overlapping character mentions only.
        # Never flatten discontinuous spans into a different mention. Reject unsupported docs.
        occupied = set()
        bad = False
        for span in doc['spans']:
            if sorted(span) != list(range(min(span), max(span) + 1)) or occupied.intersection(span):
                bad = True
            occupied.update(span)
        if bad:
            rejected.append({'doc_id': did, 'reason': 'kit_cannot_represent_discontinuous_or_overlapping_mentions'})
            continue
        offsets = doc['offsets']
        mentions = [{'id': mid, 'start': offsets[min(span)][0], 'end': offsets[max(span)][1],
                     'surface': ' '.join(doc['tokens'][i] for i in sorted(span))}
                    for mid, span in zip(doc['mentions'], doc['spans'])]
        exported = {'doc_id': did, 'topic_id': doc['topic'], 'split': 'dev',
                    'text': ' '.join(doc['tokens']), 'mentions': mentions,
                    'pairs': [{'a': p['pair'][0], 'b': p['pair'][1], 'baseline_score': p['b'],
                               **{k: p[k] for k in ('vote_ab','vote_ba','d1_ab','d1_ba')}} for p in pairs],
                    'token_offsets': offsets, 'sentence_token_ranges': doc['sentences'],
                    'original_marked_text': doc['text']}
        pilot.check_no_gold(exported)
        baseline.append(exported)
    jsonl(out / 'baseline.jsonl', baseline)
    pilot.select(SimpleNamespace(baseline=str(out / 'baseline.jsonl'), allowed_splits='dev', seed=seed,
        n_positive=12, n_negative=0, max_per_doc=2, positive_min=.7, negative_min=.25, negative_max=.39,
        out=str(out / 'selection.json')))
    chosen = load(out / 'selection.json')['selected']
    counts = Counter(x['doc_id'] for x in chosen)
    manifest = {
        'schema': 'agent4-v0-split/v1', 'selection_uses_gold': False,
        'upstream_contains_gold': True, 'boundary_allowlist': ['doc', 'a', 'b', 'luna', 'd1'],
        'seed': seed, 'threshold_fixed': .7, 'max_pairs': 12, 'max_per_document': 2,
        'dataset_revision': selection['dataset_revision'],
        'development_ids': sorted(allowed), 'excluded_ids': sorted(excluded),
        'excluded_development_overlap': sorted(ids(selection['document_ids']) & excluded),
        'reserve_status': 'ambiguous_or_exposed_excluded_regardless_of_h1_audit',
        'eligible_high_score_pairs': sum(p['baseline_score'] >= .7 for d in baseline for p in d['pairs']),
        'format_rejections': rejected,
        'selected_pairs': len(chosen), 'quotas_by_topic': dict(Counter(x['topic_id'] for x in chosen)),
        'quotas_by_document': dict(counts), 'baseline_sha256': sha(out / 'baseline.jsonl'),
        'selection_sha256': sha(out / 'selection.json'),
        'sources': {str(p): sha(p) for p in files.values()},
        'selection_notes': 'No test/confirmation/reserve document reclassified as development. No score threshold sweep.',
    }
    save(out / 'split-manifest.json', manifest)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-root', required=True)
    p.add_argument('--reserve-register', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    result = convert(a.reference_root, a.reserve_register, a.out)
    print({k: result[k] for k in ('selected_pairs', 'eligible_high_score_pairs', 'quotas_by_topic')})


if __name__ == '__main__':
    main()
