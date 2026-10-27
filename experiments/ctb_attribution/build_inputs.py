"""Package the existing frozen 20/40 sample. No sampling or model requests.
Inference files are built from whitelisted packets and cached votes only.
Evaluation gold is an explicit separate --gold step, never passed to core.py.
"""
from pathlib import Path
import argparse
import hashlib
import json

HERE=Path(__file__).parent
DEFAULT=Path('/Users/mathiscarlesso/.local/share/openresearch/files/workere/agent3-validation-finale')

def main(args):
    out=HERE/'data';out.mkdir(exist_ok=True)
    if args.gold:
        import pyarrow as pa
        assert (HERE/'rules-freeze.json').exists(),'Freeze rules before opening relation gold'
        freeze=json.loads((HERE/'rules-freeze.json').read_text())
        assert hashlib.sha256((HERE/'core.py').read_bytes()).hexdigest()==freeze['core_sha256']
        arrow=next((Path.home()/'.cache/huggingface/datasets/Nofing___causal_time_bank-standard-eci-workere').rglob('*-train.arrow'))
        with pa.memory_map(str(arrow),'r') as f:rows=pa.ipc.open_stream(f).read_all().select(['id','mentions','relations']).to_pylist()
        gold=[[r['id'],*sorted((r['mentions'][a],r['mentions'][b]))] for r in rows for a,b in r['relations']['causes']]
        baseline=[[r['document_id'],r['a'],r['b']] for r in map(json.loads,(args.source/'h1-unordered-decisions.jsonl').read_text().splitlines()) if r['positive_frozen_OR']]
        payload={'gold':sorted(gold),'baseline':sorted(baseline),'gold_usage':'EVALUATION_ONLY',
            'source_arrow_sha256':hashlib.sha256(arrow.read_bytes()).hexdigest(),
            'frozen_rules_sha256':hashlib.sha256((HERE/'rules-freeze.json').read_bytes()).hexdigest()}
        (out/'evaluation.json').write_text(json.dumps(payload,ensure_ascii=False)+'\n')
        print('Packaged evaluation-only labels after rule freeze; no model calls.')
        return
    votes={}
    for r in map(json.loads,(args.source/'h1-ordered-decisions.jsonl').read_text().splitlines()):
        if r['source']<r['target']:
            votes[r['document_id'],r['source'],r['target']]=round(r['vote_fraction']*7)
    # The input branch has no gold fields and never opens evaluation-only outputs.
    provenance={'reference_run':'a7f7bbd5-b115-4926-850a-af6eb7f44776',
        'original_selection_sha256':hashlib.sha256((args.source/'selected-components.json').read_bytes()).hexdigest(),
        'h1_ordered_sha256':hashlib.sha256((args.source/'h1-ordered-decisions.jsonl').read_bytes()).hexdigest(),
        'new_model_calls':0}
    for split in ['development','locked_test']:
        units=[]
        for row in map(json.loads,(args.source/'blinded-packets'/f'{split}.jsonl').read_text().splitlines()):
            row['candidate_votes']=[{'pair':p,'votes':votes[(row['document_id'],*p)]} for p in row['candidate_pairs']]
            row['baseline_positive_pairs']=[v['pair'] for v in row['candidate_votes'] if v['votes']>=4]
            units.append(row)
        (out/f'{split}.json').write_text(json.dumps({'provenance':provenance,'units':units},ensure_ascii=False)+'\n')
    print('Packaged 20 development and 40 test units without relation gold.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=DEFAULT);p.add_argument('--gold',action='store_true')
    main(p.parse_args())
