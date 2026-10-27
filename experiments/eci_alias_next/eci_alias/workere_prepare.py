"""Export the frozen DEV44 public inputs and historical demonstrations, offline.

Usage from repo root: PYTHONPATH=experiments/eci_alias_next uv run --project
benchmark --locked --extra hub python -m eci_alias.workere_prepare
Gold is serialized separately for the evaluator, never into public inputs/jobs.
"""
from pathlib import Path
import hashlib
import itertools
import json
import random
import subprocess
import pyarrow as pa
import yaml
from .core import Document, compact, digest, prepare_view, choose_mentions, validate_pair_view
from .cli import write_jsonl

REV = 'e6418e8747ac1c7fecf66d56ae41fbde7fd5967a'
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
SOURCE_BRANCH = 'orx/r4-a-luna-event-semantics-dev44'


def export():
    DATA.mkdir(exist_ok=True)
    selection_bytes = subprocess.check_output(['git', 'show', SOURCE_BRANCH + ':experiments/jev/round4/selection.json'])
    selection = json.loads(selection_bytes)
    selection_ids = selection['document_ids']
    bank_ids = selection['demo_bank_ids']
    cache = Path.home() / '.cache/huggingface/datasets/Nofing___event_story_line-0.9-standard-eci-workere/default/0.0.0' / REV
    tables = [pa.ipc.open_stream(cache / f'event_story_line-0.9-standard-eci-workere-{s}.arrow').read_all() for s in ('train', 'dev')]
    # Validate that the imported bank order matches the original frozen builder.
    original_selection = json.loads(Path('experiments/jev/document-selection.json').read_text())
    excluded = set(original_selection['diagnostic_d15']) | {d['id'] for d in original_selection['confirmation']['docs']}
    original_bank = sorted(did for did in tables[0].column('id').to_pylist() if did not in excluded)
    random.Random(42).shuffle(original_bank)
    assert original_bank[:20] == bank_ids, 'historical demo bank order mismatch'
    # Model input construction sees only these public columns.
    public = {}
    for table in tables:
        for row in table.select(['id', 'text', 'tokens', 'mentions', 'spans', 'topic_id']).to_pylist():
            if row['id'] not in selection_ids:
                continue
            mids = row['mentions']
            raw = dict(doc_id=row['id'], topic=str(row['topic_id']), text=row['text'], tokens=row['tokens'],
                       mentions=mids, spans=dict(zip(mids, row['spans'])),
                       candidates=[list(x) for x in itertools.combinations(sorted(mids), 2)])
            Document.from_public_json(raw)
            # Nested and discontinuous tagged serialization must align exactly.
            from types import SimpleNamespace
            from pipelines.jev.llm import marked_text
            entities = [SimpleNamespace(id=m, token_indices=tuple(s)) for m,s in zip(mids, row['spans'])]
            assert marked_text(SimpleNamespace(tokens=row['tokens']), entities) == row['text'], row['id'] + ': tagged text/span mismatch'
            public[row['id']] = raw
    assert set(public) == set(selection_ids)
    docs = [public[did] for did in selection_ids]
    assert sum(len(d['candidates']) for d in docs) * 2 == selection['ordered_pairs']
    write_jsonl(DATA / 'public_docs_DEV44.jsonl', docs)
    # Explicit exception: only fixed, protected out-of-topic demonstration labels.
    demos = {}
    for table in tables:
        for row in table.select(['id', 'text', 'annots']).to_pylist():
            if row['id'] in bank_ids:
                demos[row['id']] = row
    assert set(demos) == set(bank_ids) and not set(demos) & set(selection_ids)
    # Reconstruct the original answer serialization exactly as LunaDocumentPredictor.
    from benchmark.data import parse_annotations
    from benchmark.labels import is_norel
    prompt_bytes = Path('prompts/esl_standard_eci_lean_coref.yaml').read_bytes()
    prompt = yaml.safe_load(prompt_bytes)
    demo_messages = {}
    per_doc_demos = {}
    for did in selection_ids:
        eligible = [demos[mid] for mid in bank_ids if mid.split('_')[0] != did.split('_')[0]]
        chosen = random.Random(f'42::{did}').sample(eligible, 7)
        per_doc_demos[did] = [d['id'] for d in chosen]
        for d in chosen:
            answer = [{'pair': f'{a},{b}', 'label': label} for a, label, b in parse_annotations(d['annots']) if not is_norel(label)]
            demo_messages[d['id']] = [
                {'role': 'user', 'content': prompt['user_template'].format(doc_text=d['text'], pair_lines='', doc_id=d['id'])},
                {'role': 'assistant', 'content': json.dumps(answer)},
            ]
    (DATA / 'frozen_demos.json').write_text(json.dumps({'messages': demo_messages, 'per_document': per_doc_demos}, indent=2) + '\n')
    (DATA / 'DEV44-selection.json').write_bytes(selection_bytes)
    # Evaluation-only export happens AFTER the public inputs and selection are fixed.
    gold = {}
    for table in tables:
        for row in table.select(['id', 'mentions', 'relations']).to_pylist():
            if row['id'] in public:
                pairs = {tuple(sorted((row['mentions'][a], row['mentions'][b])))
                         for a, b in row['relations']['causes'] if a != b}
                gold[row['id']] = {'doc_id': row['id'], 'positive_pairs': [list(p) for p in sorted(pairs)]}
    private = DATA / 'evaluator_only'
    private.mkdir(exist_ok=True)
    write_jsonl(private / 'gold_DEV44.jsonl', [gold[did] for did in selection_ids])
    manifest = {
        'kind': 'frozen_development_public_inputs_and_separate_gold', 'model_calls': 0,
        'documents': len(docs), 'unordered_candidates': sum(len(d['candidates']) for d in docs),
        'selection_uses_labels': False, 'dataset_revision': REV,
        'selection_source_commit': subprocess.check_output(['git', 'rev-parse', SOURCE_BRANCH]).decode().strip(),
        'historical_prompt_sha256': hashlib.sha256(prompt_bytes).hexdigest(),
        'frozen_demo_plan_sha256': digest(compact(per_doc_demos)),
        'note': 'Demos retain historical user+answer bytes. Gold evaluation file is never supplied to rendering.',
        'files_sha256': {str(p.relative_to(DATA)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(DATA.rglob('*')) if p.is_file() and p.name != 'input-manifest.json'},
    }
    (DATA / 'input-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k:v for k,v in manifest.items() if k != 'files_sha256'}, indent=2))


if __name__ == '__main__':
    export()
