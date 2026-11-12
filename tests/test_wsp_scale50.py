import copy
import json
import re

import pytest

from steering import wsp_combo as c
from steering import wsp_scale50 as s


def make_original_maven_jobs():
    doc = dict(id='test', text='A <e0 happened> before <e1 followed>.', pair_list=[['e0','e1']])
    bank = [dict(id=f'demo{i}', text=f'demo text {i}', answer='[]') for i in range(5)]
    prompt = dict(system='rules', user_template='Document:\n{doc_text}', demos=bank)
    return {doc['id']:doc}, c.make_reading_jobs('MAVEN', {doc['id']:doc}, prompt)


def test_maven_rebuilds_same_prompts_with_constrained_pair_format_and_no_gold():
    docs, old = make_original_maven_jobs()
    extra = dict(id='extra', text='<e0 a> <e1 b>', pair_list=[['e0','e1']], gold=['e0,e1'])
    expanded = dict(docs, extra=extra)
    jobs = s.maven_jobs(expanded, docs, old)
    assert len(jobs) == 10
    for j in jobs:
        pattern=j['schema']['items']['properties']['pair']['pattern']
        assert re.fullmatch(pattern,'e1,e29')
        assert not re.fullmatch(pattern,'e1')
        assert not re.fullmatch(pattern,'e1,e2,e3')
    changed = copy.deepcopy(expanded); changed['extra']['gold'] = ['changed']
    assert s.maven_jobs(changed, docs, old) == jobs
    damaged=copy.deepcopy(old); damaged[0]['messages'][-1]['content']+=' drift'
    with pytest.raises(ValueError,match='reproduced exactly'):
        s.maven_jobs(expanded, docs, damaged)


def test_esc_reuse_deduplicates_observations_and_rejects_conflicts(tmp_path, monkeypatch):
    source,short,out=tmp_path/'full',tmp_path/'short',tmp_path/'out'
    for p in (source,short,out):p.mkdir()
    jobs=[{'key':str(i)} for i in range(33)]
    def row(i,manifest):
        return dict(key=str(i),arm='base',manifest_id=manifest,valid=True,answer='[]',
                    thought='x',num_tokens=1,run_id='source',commit='hash')
    full=[row(i,'old') for i in range(28)]
    short_ids=list(range(15))+list(range(28,33))
    shortrows=[dict(row(i,'short'),imported_from_manifest='old') for i in range(15)]+[row(i,'short') for i in range(28,33)]
    shortrows += [dict(row(i,'short'),arm='cc') for i in short_ids]
    (source/'readings.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in full))
    (short/'readings.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in shortrows))
    monkeypatch.setattr(s,'ESC_SHORT','short')
    monkeypatch.setattr(s,'ESC_CACHE_HASHES',(c.sha(source/'readings.jsonl'),c.sha(short/'readings.jsonl')))
    monkeypatch.setattr(c,'load_bundle',lambda _:({'id':'short'},None,None,jobs[:15]+jobs[28:33]))
    s.import_esc_cache(out,source,{'id':'old'},short,jobs,jobs[:28],{'id':'new'})
    assert len(c.read_cache(out/'readings.jsonl',jobs,{'id':'new'},partial=True))==53
    assert json.loads((out/'reuse.json').read_text())['observations']==68
    s.import_esc_cache(out,source,{'id':'old'},short,jobs,jobs[:28],{'id':'new'})
    shortrows[0]['answer']='changed'
    (short/'readings.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in shortrows))
    monkeypatch.setattr(s,'ESC_CACHE_HASHES',(c.sha(source/'readings.jsonl'),c.sha(short/'readings.jsonl')))
    with pytest.raises(ValueError,match='conflicting duplicate'):
        s.import_esc_cache(out,source,{'id':'old'},short,jobs,jobs[:28],{'id':'new'})
