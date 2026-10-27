#!/usr/bin/env python3
"""Freeze public ESC inputs separately from read-only diagnostic labels. No model calls.
Run with an isolated environment providing pyarrow and pyyaml, from repository root.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import random
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.data import parse_annotations
import pyarrow.ipc as ipc
import yaml

REVISION = 'e6418e8747ac1c7fecf66d56ae41fbde7fd5967a'
SOURCE = 'orx/kb-event-centric-cause-proposer-dev10'

def git_json(path):
    return json.loads(subprocess.check_output(['git', 'show', f'{SOURCE}:{path}']))

def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

def marked(row):
    starts, ends = {}, {}
    for mid, span in zip(row['mentions'], row['spans']):
        for i in span:
            if i - 1 not in span: starts.setdefault(i, []).append(str(mid))
            if i + 1 not in span: ends.setdefault(i, []).append(str(mid))
    return ' '.join(''.join(f'<{m} ' for m in starts.get(i, [])) + t + '>' * len(ends.get(i, []))
                    for i, t in enumerate(row['tokens']))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arrow', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=Path('eci_m0_pilot/data'))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = ipc.open_stream(args.arrow).read_all().to_pylist()
    public = []
    # Explicit whitelist: no relations, coreference, causal origin or annotation fields.
    for row in rows:
        ids = list(map(str, row['mentions']))
        mentions = []
        for mid, span in zip(ids, row['spans']):
            sent = next((i for i, (a, b) in enumerate(row['sentences']) if span and a <= span[0] < b), None)
            mentions.append({'id': mid, 'sentence': sent, 'span': span})
        raw_pairs = row.get('pair_list')
        candidates = itertools.combinations(ids, 2) if raw_pairs is None else (
            (ids[int(a)], ids[int(b)]) for a, b in raw_pairs if int(a) != int(b))
        pairs = sorted({tuple(sorted(p)) for p in candidates})
        public.append({'doc_id': row['id'], 'topic': row['id'].split('_')[0], 'mentions': mentions,
                       'pairs': pairs, 'text': marked(row)})
    public.sort(key=lambda d: d['doc_id'])
    (args.out / 'public_train233.jsonl').write_text(''.join(json.dumps(d, ensure_ascii=False) + '\n' for d in public))
    selection = git_json('experiments/jev/round4/selection.json')
    dev_ids = set(selection['document_ids'])
    eligible = sorted((d for d in public if d['doc_id'] in dev_ids and len(d['pairs']) >= 14),
                      key=lambda d: (len(d['pairs']), d['doc_id']))
    # Four public candidate-count strata, two documents per stratum, distinct topics.
    rng = random.Random(42)
    strata = [eligible[i * len(eligible) // 4:(i + 1) * len(eligible) // 4] for i in range(4)]
    chosen, topics = [], set()
    for stratum in strata:
        pool = list(stratum); rng.shuffle(pool)
        for d in pool:
            if d['topic'] not in topics:
                chosen.append(d['doc_id']); topics.add(d['topic'])
                if sum(x in {e['doc_id'] for e in stratum} for x in chosen) == 2: break
    if len(chosen) != 8: raise ValueError('unable to select eight distinct-topic development documents')
    # Interleave strata so the initial two-document smoke spans candidate scales.
    chosen = chosen[::2] + chosen[1::2]
    bank_ids = selection['demo_bank_ids']
    historic = git_json('experiments/jev/document-selection.json')
    excluded = set(historic['diagnostic_d15']) | {d['id'] for d in historic['confirmation']['docs']}
    bank = sorted((r for r in rows if r['id'] not in excluded), key=lambda r: r['id'])
    random.Random(42).shuffle(bank)
    assert [r['id'] for r in bank[:20]] == bank_ids, 'historical demo bank provenance mismatch'
    demos = []
    for row in bank[:20]:
        demos.append({'doc_id': row['id'], 'topic': row['id'].split('_')[0], 'text': row['text'],
                      'mentions': [{'id': str(x)} for x in row['mentions']],
                      'answer': [{'pair': f'{a},{b}', 'label': lab} for a, lab, b in parse_annotations(row['annots'])
                                 if lab != 'norel']})
    save(args.out / 'demonstrations.json', demos)
    source_prompt = Path('prompts/esl_standard_eci_lean_coref.yaml')
    save(args.out / 'prompt.json', yaml.safe_load(source_prompt.read_text()))
    schedule = {}
    for did in chosen:
        pool = [d for d in demos if d['topic'] != did.split('_')[0]]
        schedule[did] = [d['doc_id'] for d in random.Random(f'42::{did}').sample(pool, 7)]
        assert did not in schedule[did]
    save(args.out / 'selection.json', {'selection_uses_labels': False, 'seed': 42,
         'status': 'consumed DEV44, exploratory, not a fresh confirmation',
         'rule': 'four candidate-count strata; two documents per stratum; distinct topics; deterministic seed 42',
         'document_ids': chosen, 'smoke_document_ids': [chosen[0], chosen[3]],
         'demonstration_schedule': schedule, 'demo_bank_ids': bank_ids})
    # Gold first becomes accessible here, after the public selection and schedule are frozen.
    controls = git_json('experiments/jev/round4/controls.json')
    grouped = {}
    for r in controls:
        if r['doc'] in chosen:
            p = tuple(sorted((r['a'], r['b'])))
            grouped.setdefault((r['doc'], p), []).append(r)
    evaluation = []
    for (did, p), group in sorted(grouped.items()):
        assert len(group) == 2 and {(x['a'], x['b']) for x in group} == {p, p[::-1]}
        score = sum(float(x['d1']) + float(x['luna']) for x in group) / 4
        evaluation.append({'doc_id': did, 'pair': p, 'gold': any(x['gold'] for x in group),
                           'baseline_positive': score >= .39, 'baseline_score': score,
                           'luna_zero': all(float(x['luna']) == 0 for x in group)})
    save(args.out / 'evaluation_only.json', evaluation)
    for d in public:
        if d['doc_id'] in chosen:
            assert {tuple(x['pair']) for x in evaluation if x['doc_id'] == d['doc_id']} == set(map(tuple, d['pairs']))
    save(args.out / 'provenance.json', {'dataset_revision': REVISION,
         'arrow_sha256': hashlib.sha256(args.arrow.read_bytes()).hexdigest(),
         'control_source_commit': subprocess.check_output(['git', 'rev-parse', SOURCE]).decode().strip(),
         'prompt_sha256': hashlib.sha256(source_prompt.read_bytes()).hexdigest(),
         'gold_access': 'evaluation_only.json; demonstration answers only for frozen outside-topic examples',
         'files_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(args.out.iterdir()) if p.is_file() and p.name != 'provenance.json'}})
    print(json.dumps({'public_docs': len(public), 'selected': chosen, 'pairs': len(evaluation), 'model_calls': 0}))

if __name__ == '__main__': main()
