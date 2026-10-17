"""Validate Luna's frozen confirmation identity before constructing model clients."""
import hashlib
from pathlib import Path
from benchmark.configuration import read_yaml, load_suite
from benchmark.io import read_json


def validate(path, configs):
    path=Path(path).resolve()
    campaign=read_yaml(path)
    proof=campaign['confirmation_freeze']
    loaded={}
    for key in ('freeze','suite'):
        source=path.parent/proof[key]
        if hashlib.sha256(source.read_bytes()).hexdigest()!=proof[key+'_sha256']:
            raise ValueError('sealed confirmation '+key+' hash mismatch')
        loaded[key]=read_json(source)
    frozen,canonical=loaded['freeze'],loaded['suite']
    expected={d['id']:d for d in canonical['documents']}
    if (set(expected)!=set(frozen['confirmation_document_ids']) or len(expected)!=40
            or sum(len(d['pair_list']) for d in expected.values())!=19174
            or campaign['max_parallel']!=8 or len(configs)!=40):
        raise ValueError('confirmation universe or bounded concurrency differs')
    seen=set()
    for config in configs:
        cfg=config.data
        params=cfg['pipeline']['relation_predictor']['params'];client=params['client']
        if (cfg['dataset']['split']!='train' or cfg['seed']!=42 or cfg['protocol']!='relations_gold'
                or cfg['pipeline']['relation_predictor']['plugin']!='luna_document'
                or client['model']!=frozen['llm_reference']['model']
                or client['reasoning_effort']!=frozen['llm_reference']['reasoning_effort']
                or params['passes']!=frozen['llm_reference']['passes']
                or params['revision']!=frozen['dataset_revision']
                or client['max_output_tokens']!=32768
                or client['shared_budget']['scope']!=frozen['global_budget_scope']
                or client['shared_budget']['limit_usd']!=frozen['global_budget_usd']):
            raise ValueError('Luna scientific settings differ from frozen reference')
        if hashlib.sha256(Path(params['path']).read_bytes()).hexdigest()!=frozen['llm_reference']['prompt_sha256']:
            raise ValueError('Luna prompt differs from frozen reference')
        selection=path.parents[2]/params['selection_path']
        if hashlib.sha256(Path(selection).read_bytes()).hexdigest()!=frozen['selection_manifest_sha256']:
            raise ValueError('demonstration exclusion manifest differs')
        suite=load_suite(config)
        if len(suite['documents'])!=1:
            raise ValueError('one document required per bounded batch')
        doc=suite['documents'][0];did=doc['id']
        if did in seen or expected.get(did)!=doc:
            raise ValueError('duplicate document or candidate/label mismatch')
        seen.add(did)
    if seen!=set(expected):
        raise ValueError('incomplete confirmation universe')
