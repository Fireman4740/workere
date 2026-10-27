"""No-gold V0 scoring wrapper: frozen kit contrast plus explicit directional scores.
Refuses empty/unsealed/unreviewed runs. Does not expose a gold/evaluate command.
"""
import argparse
from collections import defaultdict, Counter
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
from .common import rows, require, jsonl, save, sha
from .adapter import verify_review
from .kit import pilot


def score(tasks, predictions, baseline, selection, certificate, out):
    out=Path(out)
    require(not out.exists(), 'scoring output exists; do not overwrite a lock')
    ts, ps=rows(tasks),rows(predictions)
    require(bool(ts) and bool(ps), 'no actual task/output set; do not manufacture empty results')
    verify_review(tasks,baseline,selection,certificate)
    require(all(p.get('origin') != 'MOCK_NOT_A_MODEL' for p in ps), 'MOCK data cannot be a real V0 result')
    require(len({t['task_id'] for t in ts}) == len(ts), 'duplicate task')
    counts=Counter((t['key'],t['variant']) for t in ts)
    expected={'original':2,'target':1,'placebo_1':1,'placebo_2':1,'placebo_3':1,'rename_ids':1}
    for key in {t['key'] for t in ts}:
        require({v:n for (k,v),n in counts.items() if k==key} == expected, 'not V0 read allocation')
    out.mkdir(parents=True)
    pilot.score(SimpleNamespace(tasks=str(tasks),predictions=str(predictions),out=str(out/'scores.jsonl')))
    by_id={p['task_id']:p for p in ps}
    directional=defaultdict(lambda:defaultdict(list))
    for t in ts:
        p=by_id.get(t['task_id'])
        if p is None or p.get('status')!='ok':
            directional[t['key']][t['variant']].append(None)
            continue
        a,b=t['exposed_pair']
        pairs={tuple(x) for x in p['predicted_ordered_pairs']}
        directional[t['key']][t['variant']].append((int((a,b) in pairs),int((b,a) in pairs)))
    output=[]
    for key,conditions in directional.items():
        summary={}
        for variant,votes in conditions.items():
            summary[variant]=None if any(v is None for v in votes) else {
                'vote_ab':mean(v[0] for v in votes),'vote_ba':mean(v[1] for v in votes), 'reads':len(votes)}
        output.append({'key':key,'directional_scores':summary,'interpretation':'empirical votes, not gold probabilities'})
    jsonl(out/'directional-scores.jsonl',output)
    save(out/'freeze-before-any-gold.json',{'status':'LOCK_BEFORE_GOLD_NO_F1',
        'files_sha256':{str(p):sha(p) for p in (tasks,predictions,baseline,selection,certificate,
            out/'scores.jsonl',out/'scores.jsonl.lock.json',out/'directional-scores.jsonl')},
        'warning':'1–2 reads/condition: coarse feasibility descriptors only, no final F1.'})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('tasks','predictions','baseline','selection','certificate','out'):
        p.add_argument('--'+name,required=True)
    a=p.parse_args()
    score(a.tasks,a.predictions,a.baseline,a.selection,a.certificate,a.out)


if __name__ == '__main__':
    main()
