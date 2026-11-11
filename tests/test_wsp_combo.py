import copy
import json

import pytest

from steering import wsp_combo as c


def test_final_parser_rejects_incomplete_contradictory_and_duplicate_keys():
    assert c.parse_document('[]') == set()
    assert c.parse_document('[{"pair":"a,b","label":"causes"}]') == {('a', 'b')}
    assert c.parse_document('[{"pair":"a,b","label":"causes"},{"pair":"b,a","label":"norel"}]') is None
    assert c.parse_document('[{"pair":"a,b","label":"causes","label":"norel"}]') is None
    assert c.parse_document('reasoning []') is None
    assert c.parse_document('[{"pair":"a,a","label":"causes"}]') is None
    assert c.parse_block('{"0":"N","0":"A"}', 1) is None
    assert c.parse_block('{"0":"N"}', 2) is None


def test_fixed_denominator_never_promotes_one_valid_vote():
    p = ('a', 'b')
    assert c.vote([{p}, None, None, None, None]) == set()
    assert c.vote([{p}, {p}, {p}, None, None]) == {p}
    assert c.vote([{p}, set()]) == {p}  # predeclared tie-positive historical rule


def test_demo_planning_never_reads_gold_or_same_topic():
    doc = dict(id='1_test', text='<a fall> <b break>', pair_list=[['a','b']])
    demos = [dict(id=f'2_{i}', topic='2', text=f'demo {i}', answer='[]') for i in range(5)]
    demos += [dict(id='1_same', topic='1', text='same topic', answer='[]'),
              dict(id='1_test', topic='2', text='different text', answer='[]')]
    prompt = dict(system='rules', user_template='{doc_text}', demos=demos)
    jobs = c.make_reading_jobs('ESC', {doc['id']:doc}, prompt)
    changed = dict(doc, gold=['leaked'], chains=[['a','b']])
    assert jobs == c.make_reading_jobs('ESC', {doc['id']:changed}, prompt)
    assert len(jobs) == 5
    assert len({j['meta']['demo_id'] for j in jobs}) == 5
    assert all(j['meta']['demo_id'].startswith('2_') for j in jobs)
    assert all(j['key'] == c.digest({k:v for k,v in j.items() if k!='key'}) for j in jobs)


def test_routing_capped_without_gold_and_filters_outside_candidates():
    ps = {tuple(sorted(('a', str(i)))) for i in range(30)}
    doc = dict(id='d', text='source', pair_list=list(ps))
    readings = [ps | {('ghost','a')}, set(), ps, set(), ps, set(), ps, set(), ps, set()]
    systems = dict(base=set(), cc=ps)
    a, count = c.routed_pairs(doc, readings, systems)
    assert len(a) == c.ROUTE_CAP and count == 30
    assert set(a) <= ps
    assert (a, count) == c.routed_pairs(dict(doc,gold=['changed']), readings, systems)


def test_two_event_orders_require_direction_agreement_and_valid_answers():
    jobs=[]
    for reverse in (False,True):
        jobs.append(c.job(dict(kind='verification',doc_id='d',reverse=reverse,pairs=[['a','b']]), [],c.block_schema(1),4096,512))
    rows={(a,j['key']):dict(valid=True,answer=json.dumps({'0':'B' if j['meta']['reverse'] else 'A'}))
          for a in ('base','cc') for j in jobs}
    result=c.judge_decisions(jobs,rows)[('d',('a','b'))]
    assert result==dict(base='A',cc='A',joint='A')
    rows[('cc',jobs[1]['key'])]['valid']=False
    result=c.judge_decisions(jobs,rows)[('d',('a','b'))]
    assert result==dict(base='A',cc=None,joint=None)
    rows[('base',jobs[1]['key'])]['answer']='{"0":"A"}'
    assert c.judge_decisions(jobs,rows)[('d',('a','b'))]['base'] is None


def test_policy_abstentions_and_additions_only_do_not_delete():
    p,q=('a','b'),('c','d')
    base={'doc':{p}}
    decisions={('doc',p):dict(base='N',cc=None,joint=None),
               ('doc',q):dict(base='A',cc='A',joint='A')}
    assert c.combine(base,decisions,'base_filter')=={'doc':set()}
    assert c.combine(base,decisions,'base_repair')=={'doc':{q}}
    assert c.combine(base,decisions,'base_add')=={'doc':{p,q}}
    assert c.combine(base,decisions,'joint_repair')=={'doc':{p,q}}
    assert c.combine(base,decisions,'strict_add_base_filter')=={'doc':{q}}
    assert base=={'doc':{p}}


def test_missing_and_foreign_caches_are_never_scored(tmp_path):
    path=tmp_path/'rows.jsonl';plan={'id':'version'};jobs=[{'key':'a'}]
    path.write_text(json.dumps(dict(arm='base',key='a',manifest_id='version'))+'\n')
    with pytest.raises(ValueError,match='incomplete'):c.read_cache(path,jobs,plan)
    assert len(c.read_cache(path,jobs,plan,partial=True))==1
    path.write_text(json.dumps(dict(arm='base',key='a',manifest_id='wrong'))+'\n')
    with pytest.raises(ValueError,match='stale'):c.read_cache(path,jobs,plan,partial=True)


def test_pooling_uses_actual_historical_support_not_aggregate_repetition():
    p=('a','b')
    groups={'d':{'base':[set()]*5,'cc':[{p}]*5}}
    prior=dict(predictions={'d':[]},support={'d':{'a,b':4}},valid_readings={'d':10})
    systems=c.vote_systems(groups,prior)
    assert systems['historical_plus_cc5']['d']=={p}  # 9/15
    assert systems['historical_plus_base5']['d']==set()  # 4/15
    assert systems['historical']['d']==set()


def test_scoring_keeps_negative_docs_and_outside_penalty_separate():
    docs={'yes':dict(id='yes',text='',pair_list=[['a','b'],['b','a']],gold=['a,b','b,a']),
          'no':dict(id='no',text='',pair_list=[['a','b'],['b','a']],gold=[])}
    predictions={'yes':{('a','b'),('ghost','other')},'no':{('a','b')}}
    r=c.evaluate('MAVEN',docs,{'base5':predictions})['base5']
    assert (r['tp'],r['fp'],r['fn'],r['documents'])==(2,2,0,2)
    assert r['outside_candidate_predictions']==2
    assert r['f1_with_outside_penalty']==.5


def test_topic_bootstrap_uses_topics_not_documents():
    per={'1_a':dict(tp=2,fp=0,fn=0),'1_b':dict(tp=0,fp=2,fn=2),'2_a':dict(tp=2,fp=0,fn=0)}
    r=c.paired(per,copy.deepcopy(per),'ESC')
    assert r['units']==2 and r['ci95_pp']==[0.,0.] and r['delta_f1_pp']==0.


def test_freeze_resumes_tuple_pairs_but_rejects_changed_science(tmp_path):
    p=tmp_path/'plan.json'
    c.freeze(p,dict(pairs=[('a','b')]))
    c.freeze(p,dict(pairs=[('a','b')]))
    with pytest.raises(ValueError,match='frozen'):c.freeze(p,dict(pairs=[('a','c')]))
