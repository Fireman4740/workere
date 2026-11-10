"""CPU-only, preregistered combinations of original cached method decisions."""
from collections import Counter, defaultdict
import copy
import os
from pathlib import Path
import time

from steering import ap_campaign as ap
from steering import ap_protocol as p
from steering import campaign_api as api
from steering import campaign_blocks as c
from steering import validation_protocol as v

VERSION = 'method-vote-20261110-v1'
METHODS = ('screen_majority_control', 'five_view_majority', 'four_view_consensus',
           'strict_add_majority_remove')


def combine(base, candidates, views, judgments):
    """No gold accepted. Missing checks abstain; incomplete panels retain base."""
    judges = defaultdict(dict)
    for row in judgments:
        key = (row['dataset'], row['doc_id'], tuple(sorted(row['pair'])))
        assert row['reverse'] not in judges[key], 'duplicate adjudicator orientation'
        label = row.get('label') if row['status'] == 'valid' else None
        assert label in (None, 'A', 'B', 'N')
        judges[key][row['reverse']] = None if label is None else label != 'N'
    systems = {name: copy.deepcopy(base) for name in METHODS}
    coverage = Counter()
    disagreement = Counter()
    for ds, documents in candidates.items():
        for doc, pairs in documents.items():
            for pair in pairs:
                pair = tuple(sorted(pair))
                original = pair in base[ds][doc]
                screen = [views.get((ds, doc, o), {}).get(pair) for o in (0, 1)]
                judge = [judges.get((ds, doc, pair), {}).get(o) for o in (False, True)]
                checks = screen + judge
                coverage['candidate_pairs'] += 1
                coverage['screen_complete'] += None not in screen
                coverage['five_view_complete'] += None not in checks
                for name, values in [('screen', screen), ('judge', judge)]:
                    if None not in values:
                        disagreement[name+'_paired'] += 1
                        disagreement[name+'_disagree'] += values[0] != values[1]
                decisions = dict.fromkeys(METHODS, original)
                if None not in screen:
                    decisions['screen_majority_control'] = sum(screen)+original >= 2
                if None not in checks:
                    positives = sum(checks)
                    decisions['five_view_majority'] = positives+original >= 3
                    decisions['four_view_consensus'] = (positives == 4 if not original else positives != 0)
                    decisions['strict_add_majority_remove'] = (positives == 4 if not original else positives > 1)
                for name, present in decisions.items():
                    if present:
                        systems[name][ds][doc].add(pair)
                    else:
                        systems[name][ds][doc].discard(pair)
    return systems, dict(coverage), dict(disagreement)


def load_predictions(path):
    return {ds: {d: {tuple(pair) for pair in pairs} for d, pairs in docs.items()}
            for ds, docs in ap.read(path).items()}


def score_dataset(ds, source, out):
    terminal = ap.read(source/'terminal.json')
    assert terminal['status'] == 'complete'
    previous = ap.read(source/'report.json')
    assert c.sha(source/'report.json') == terminal['report_sha256']
    selected = previous['selected'][ds]
    if ds == 'MAVEN':
        docs = ap.read(source/'documents.json')
    else:
        ctb, _, esc, _ = c.inputs()
        docs = {ds: ctb if ds == 'CTB' else esc}
    candidates = {ds: {d: c.pairs(docs[ds][d]) for d in selected}}
    base = load_predictions(source/'A_vote.json')
    repair = load_predictions(source/'C_repair.json')
    assert set(base[ds]) == set(selected) == set(repair[ds])
    jobs = c.jsonl(source/'screen/requests.jsonl')
    outputs = {r['key']: r for r in c.jsonl(source/'screen/responses.jsonl')}
    assert {j['key'] for j in jobs} <= set(outputs), 'incomplete source screen'
    judgments = c.jsonl(source/'repair/judgments.jsonl')
    systems, coverage, agreement = combine(base, candidates, v.views_from(jobs, outputs), judgments)
    systems = dict(A_vote=base, C_repair=repair, **systems)
    # Reproduce archived scorer counts before evaluating any new policy.
    metrics = {name: p.metrics(pred, base, docs)[ds] for name, pred in systems.items()}
    for name in ('A_vote', 'C_repair'):
        assert metrics[name]['per_document'] == previous['policies'][name]['per_document']
    vs_repair = {name: p.metrics(pred, repair, docs)[ds]['delta'] for name, pred in systems.items()}
    paths = [source/x for x in ('report.json', 'manifest.json', 'A_vote.json', 'C_repair.json',
             'screen/requests.jsonl', 'screen/responses.jsonl', 'repair/judgments.jsonl')]
    manifest = dict(version=VERSION, methods=METHODS, selected=selected,
        sources={str(path): c.sha(path) for path in paths},
        missing='abstain; retain baseline for incomplete required panel',
        independence='same model, correlated presentations; no independent-voter assumption',
        scope='reused development; exploratory fixed-policy comparison, no held-out claim')
    out.mkdir(parents=True, exist_ok=True)
    ap.freeze(out/'manifest.json', manifest)
    for name, pred in systems.items():
        ap.dump_predictions(out/(name+'.json'), pred)
    # Compare identical prediction sets, not just identical aggregate F1.
    aliases = [(n, m) for i, n in enumerate(systems) for m in list(systems)[i+1:]
               if systems[n] == systems[m]]
    report = dict(dataset=ds, policies=metrics, vs_repair=vs_repair, coverage=coverage,
        agreement=agreement, exact_prediction_duplicates=aliases,
        source_run=previous['run_id'], selected=selected, manifest=manifest,
        new_model_calls=0, upstream_cost=previous.get('cost'),
        cost_note='CPU recombination only; all upstream inference remains part of deployment cost',
        run_id=os.environ.get('ORX_RUN_ID'), commit=os.environ.get('ORX_COMMIT'), status='complete')
    ap.save(out/'report.json', report)
    print('MV_RESULT', api.canonical(report), flush=True)
    return report


def main():
    work = Path(os.environ['WORK'])
    source = work/'out/VC'/os.environ['VC_GROUP']
    out = work/'out/MV'/os.environ['MV_GROUP']
    out.mkdir(parents=True, exist_ok=True)
    ap.freeze(out/'design.json', dict(version=VERSION, methods=METHODS,
        datasets=['CTB', 'ESC', 'MAVEN'], inference=False,
        selection='report every fixed policy; no gold-conditioned per-document winner'))
    print('MV_CONFIG', api.canonical(ap.read(out/'design.json')), flush=True)
    reports = {}
    for ds in ('CTB', 'ESC', 'MAVEN'):
        if ds == 'MAVEN':
            expected = os.environ['MV_MAVEN_RUN']
            deadline = time.time()+13*3600
            while True:
                terminal = source/ds/'terminal.json'
                state = ap.read(terminal) if terminal.exists() else {}
                if state.get('status') == 'complete':
                    assert ap.read(source/ds/'report.json')['run_id'] == expected
                    break
                if (source/ds/f'wall-{expected}.json').exists() and state.get('status') in ('failed','budget_exhausted'):
                    raise RuntimeError('MAVEN upstream failed; CTB/ESC reports retained, no partial MAVEN score')
                if time.time() > deadline:
                    raise TimeoutError('MAVEN upstream not complete; CTB/ESC reports retained')
                print('MV_WAIT MAVEN upstream', expected, flush=True)
                time.sleep(30)
        reports[ds] = score_dataset(ds, source/ds, out/ds)
    ap.save(out/'report.json', reports)
    print('MV_COMPLETE', str(out/'report.json'), flush=True)


if __name__ == '__main__':
    main()
