import copy
import json

import pytest

from steering import wsp_combo as c
from steering import wsp_short as s


def fixture_bundle(dataset='ESC'):
    ids = ['12_a', '12_b', '13_a', '13_b', '14_a', '16_a']
    subsets = {} if dataset == 'ESC' else dict(original12=ids[:3], extension12=ids[3:])
    docs = {d: dict(id=d, text=d, pair_list=[['a', 'b'], ['b', 'a']], gold=['a,b', 'b,a']) for d in ids}
    jobs = [c.job(dict(doc_id=d, pass_id=p, kind='reading'), [dict(role='user', content=d)],
                  c.document_schema(), 8192, 4096) for d in ids for p in range(5)]
    prior = dict(predictions={d: [['a', 'b']] for d in ids}, support={d: {'a,b': 3} for d in ids},
                 valid_readings={d: 5 for d in ids}, emitted_tokens={d: 123 for d in ids},
                 invalid={d: 0 for d in ids}, source_sha256={'old': 'hash'}, scope='source')
    parent = dict(id='parent', dataset=dataset, documents=ids, subsets=subsets)
    return parent, docs, prior, jobs


def test_four_documents_distinct_topics_or_balanced_cohorts():
    parent, _, _, _ = fixture_bundle()
    ids, groups, _ = s.select_ids('ESC', parent['documents'], {})
    assert ids == ['12_a', '13_a', '14_a', '16_a'] and groups == {}
    assert len({d.split('_')[0] for d in ids}) == 4
    parent, _, _, _ = fixture_bundle('MAVEN')
    ids, groups, _ = s.select_ids('MAVEN', parent['documents'], parent['subsets'])
    assert groups == dict(original2=['12_a', '12_b'], extension2=['13_b', '14_a'])
    assert ids == sorted(sum(groups.values(), []))
    with pytest.raises(ValueError, match='four distinct'):
        s.select_ids('ESC', ['1_a', '1_b'], {})


def test_projection_preserves_exact_requests_filters_all_history_and_ignores_gold():
    parent, docs, prior, jobs = fixture_bundle()
    before = copy.deepcopy((docs, prior, jobs))
    selected, history, requests, _, _ = s.project(parent, docs, prior, jobs)
    assert len(requests) == 20
    assert requests == [j for j in jobs if j['meta']['doc_id'] in selected]
    for field in s.PER_DOCUMENT_FIELDS:
        assert history[field] == {d: prior[field][d] for d in selected}
    changed = copy.deepcopy(docs)
    for row in changed.values():
        row['gold'] = ['unrelated']
    new = s.project(parent, changed, prior, jobs)
    assert set(new[0]) == set(selected) and new[2:] == (requests, {}, s.select_ids('ESC', parent['documents'], {})[2])
    assert (docs, prior, jobs) == before
    jobs[0]['messages'][0]['content'] = 'changed'
    with pytest.raises(ValueError, match='fingerprint'):
        s.project(parent, docs, prior, jobs)


def test_import_preserves_invalid_observations_cost_and_provenance_and_resumes(tmp_path, monkeypatch):
    parent, docs, prior, jobs = fixture_bundle()
    _, _, selected_jobs, _, _ = s.project(parent, docs, prior, jobs)
    source, out = tmp_path/'source', tmp_path/'out'
    source.mkdir(); out.mkdir()
    row = dict(key=jobs[0]['key'], arm='base', manifest_id='parent', valid=False,
               answer='invalid', thought='trace', thinking_tokens=7, answer_tokens=1,
               num_tokens=8, run_id='old-run', commit='old-commit')
    excluded = dict(row, key=jobs[5]['key'])  # second document in the same topic
    path = source/'readings.jsonl'
    path.write_text(''.join(json.dumps(r)+'\n' for r in [row, excluded]))
    monkeypatch.setitem(s.CACHE_SHA256, 'ESC', c.sha(path))
    plan = dict(id='short')
    s.import_cache(source, out, parent, plan, jobs, selected_jobs)
    assert json.loads((out/'imported_source_rows.json').read_text()) == [row]
    imported = json.loads((out/'readings.jsonl').read_text())
    assert imported == dict(row, manifest_id='short', imported_from_manifest='parent')
    assert json.loads((out/'reuse.json').read_text())['tokens'] == 8
    fresh = dict(imported, key=selected_jobs[1]['key'], run_id='new-run')
    with (out/'readings.jsonl').open('a') as f:
        f.write(json.dumps(fresh)+'\n')
    original = (out/'readings.jsonl').read_bytes()
    s.import_cache(source, out, parent, plan, jobs, selected_jobs)
    assert (out/'readings.jsonl').read_bytes() == original
    imported['answer'] = 'changed'
    (out/'readings.jsonl').write_text(json.dumps(imported)+'\n'+json.dumps(fresh)+'\n')
    with pytest.raises(ValueError, match='observation changed'):
        s.import_cache(source, out, parent, plan, jobs, selected_jobs)


@pytest.mark.parametrize('failure', ['hash', 'manifest', 'duplicate'])
def test_import_rejects_drift_stale_or_duplicate_source(tmp_path, monkeypatch, failure):
    parent, docs, prior, jobs = fixture_bundle()
    source, out = tmp_path/'source', tmp_path/'out'
    source.mkdir(); out.mkdir()
    row = dict(key=jobs[0]['key'], arm='base', manifest_id='wrong' if failure=='manifest' else 'parent')
    path = source/'readings.jsonl'
    path.write_text((json.dumps(row)+'\n') * (2 if failure=='duplicate' else 1))
    monkeypatch.setitem(s.CACHE_SHA256, 'ESC', 'wrong' if failure=='hash' else c.sha(path))
    with pytest.raises(ValueError, match='cache changed|duplicate, foreign, or stale'):
        s.import_cache(source, out, parent, dict(id='short'), jobs, jobs)
    assert not (out/'readings.jsonl').exists()
