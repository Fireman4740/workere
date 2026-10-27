"""Local M1 invariants and conservative budget audit; synthetic labels only."""
import argparse
from collections import Counter
import copy
import importlib.util
import itertools
import json
from pathlib import Path
import subprocess
import sys
import unittest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent))
from prepare import requests, SETTINGS, sha, save
from eci_m0_pilot.m0 import as_pairs,build,pair
from eci_m0_pilot.protocol import decode_response,remap_tags,lexical_text,TAG,json_text,coverage
spec=importlib.util.spec_from_file_location('m1_analysis',HERE/'m1-analysis.py')
analysis=importlib.util.module_from_spec(spec);spec.loader.exec_module(analysis)

def toy(n=9):
    ids=list(map(str,range(1,n+1)))
    return dict(doc_id='toy',topic='t',mentions=[dict(id=x,sentence=int(x)%3,span=[int(x)]) for x in ids],
        pairs=list(itertools.combinations(ids,2)),text='2017 12 people. '+' '.join(f'<{x} event>' for x in ids))
def response(s='[]',finish='stop'):
    return {'response':{'status_code':200,'body':{'model':SETTINGS['model'],'choices':[{'finish_reason':finish,'message':{'content':s}}]}}}

class Tests(unittest.TestCase):
    def test_canonicalization(self):
        self.assertEqual(pair('2','1'),pair('1','2'))
        with self.assertRaises(ValueError):pair('1','1')
    def test_partition_permutation_and_text(self):
        for n in (2,3,8,21):
            d=toy(n);rr=build(d,7,SETTINGS['seed']);u=as_pairs(d['pairs'])
            mp=[r for r in rr if r['arm']=='MP'];cnt=Counter(pair(*p) for r in mp for p in r['allowed_original'])
            self.assertEqual(set(cnt),u);self.assertEqual(set(cnt.values()),{1})
            for r in mp:
                m=r['id_mapping'];self.assertEqual(set(m),set(m.values()))
                t=remap_tags(d['text'],m);self.assertEqual(lexical_text(t),lexical_text(d['text']))
                self.assertEqual(remap_tags(t,{v:k for k,v in m.items()}),d['text'])
                self.assertEqual(as_pairs(r['allowed_visible']),{pair(m[a],m[b]) for a,b in r['allowed_original']})
    def test_empty_invalid_unknown_reject_and_offmask(self):
        r=build(toy(3))[0];r['allowed_original']=[['1','2']]
        self.assertEqual(decode_response(r,response(),SETTINGS)['state'],'VALID_EMPTY')
        for s in ('garbage','[{"pair":"1,999","label":"causes"}]','[7]'):
            self.assertFalse(decode_response(r,response(s),SETTINGS)['valid'])
        self.assertEqual(decode_response(r,response('[]','length'),SETTINGS)['state'],'INCOMPLETE')
        v=decode_response(r,response('[{"pair":"1,3","label":"causes"},{"pair":"1,2","label":"norel"}]'),SETTINGS)
        self.assertTrue(v['valid']);self.assertEqual(v['positive'],[])
        self.assertEqual(v['explicit_reject'],[('1','2')]);self.assertEqual(v['outside_mask'],[('1','3')])
    def test_inversion_before_deduplication(self):
        r=build(toy(3))[0];r['id_mapping']={'1':'3','2':'1','3':'2'}
        v=decode_response(r,response('[{"pair":"3,1","label":"causes"},{"pair":"1,3","label":"causedby"}]'),SETTINGS)
        self.assertEqual(v['positive'],[('1','2')]);self.assertEqual(v['duplicates_removed'],1)
    def test_synthetic_metrics_and_complementarity(self):
        a,b,c,d,e=[('doc',str(i),str(i+1)) for i in range(5)]
        z=analysis.evaluate({a,b,c},{a,d},{'O_shared':{b},'O_extra':{e},'MP':{c}},{a,b,c,d,e},{'doc':'topic'})
        self.assertEqual(z['systems']['C']['tp'],2);self.assertEqual(z['systems']['C']['fp'],2)
        self.assertEqual(z['systems']['H']['tp'],3);self.assertEqual(z['systems']['H']['fp'],1)
        self.assertAlmostEqual(z['systems']['H']['f1'],6/7)
        self.assertEqual(z['conditional_MP']['ordinary_extra_missed_tp'],1)
        self.assertEqual(z['categories']['001']['tp'],1)
        self.assertEqual(z['bootstrap']['topics'],1)

def audit(dest,hashseed_child=False):
    p=json.loads((dest/'m1-protocol-frozen.json').read_text());cfg=p['settings']
    docs=json.loads((dest/'m1-public-documents.json').read_text())
    rows=[json.loads(l) for l in (dest/'m1-requests.jsonl').read_text().splitlines()]
    demos=json.loads((HERE.parent/'eci_m0_pilot/data/demonstrations.json').read_text())
    template=json.loads((HERE.parent/'eci_m0_pilot/data/prompt.json').read_text())
    checks={};doc_checks={}
    for did,d in docs.items():
        rr=[r for r in rows if r['doc_id']==did]
        rebuilt,schedule=requests(d,demos,template,cfg)
        assert json_text(rr)==json_text(rebuilt)
        assert schedule==p['demonstration_schedule'][did]
        assert Counter(r['channel'] for r in rr)=={'O_shared':7,'O_extra':8,'MP':7}
        u=as_pairs(d['pairs']);mp=[r for r in rr if r['channel']=='MP']
        cnt=Counter(pair(*pair_) for r in mp for pair_ in r['allowed_original'])
        assert set(cnt)==u and set(cnt.values())=={1}
        for r in rr:
            assert r['demo_id']!=did and r['demo_id'].split('_')[0]!=d['topic']
            assert r['prompt_sha256']==sha(json_text(r['messages']).encode())
            assert r['body']['messages']==r['messages']
            rendered=remap_tags(d['text'],r['id_mapping'])
            assert rendered in r['messages'][-1]['content']
            assert lexical_text(rendered)==lexical_text(d['text'])
            assert set(r['id_mapping'])==set(r['id_mapping'].values())=={str(m['id']) for m in d['mentions']}
            assert as_pairs(r['allowed_original'])<=u
        sentence={str(m['id']):m['sentence'] for m in d['mentions']}
        intra=sum(sentence[a]==sentence[b] for a,b in u)
        doc_checks[did]=dict(candidate_pairs=len(u),intra=intra,inter=len(u)-intra,
            MP_unique_exposure=True,block_sizes=[len(r['allowed_original']) for r in mp],
            per_mention={i:dict(candidate_degree=sum(i in x for x in u),
                planned_incident_pair_exposures=sum(i in x for r in mp for x in r['allowed_original']),
                masks_with_incident_pair=sum(any(i in x for x in r['allowed_original']) for r in mp)) for i in sentence})
    assert len(rows)==p['total_unique_requests']<=528
    assert sha(dest/'m1-requests.jsonl')==p['provenance']['requests_sha256']
    for name,h in p['provenance']['files_sha256'].items():assert sha(dest/name)==h
    summary=dict(protocol_sha256=sha(dest/'m1-protocol-frozen.json'),requests_sha256=sha(dest/'m1-requests.jsonl'),
        documents=len(docs),topics=len({d['topic'] for d in docs.values()}),unique_requests=len(rows),
        conservative_api_ceiling_usd=p['conservative_api_ceiling_usd'],
        input_tokens_upper_bound=sum(r['prompt_utf8_upper_bound'] for r in rows),
        completion_tokens_upper_bound=len(rows)*cfg['max_output_tokens'],
        largest_prompt_token_upper_bound=max(r['prompt_utf8_upper_bound'] for r in rows),
        token_counter='serialized UTF8 bytes +1024 overhead per request, conservative upper bound; no exact tokenizer count claimed',
        inputs_gold_free=True,exact_universe_preserved=True,demonstrations_outside_topic=True,
        complete_prompts_rebuilt_and_audited=True,original_text_and_spans_preserved=True,
        per_document=doc_checks)
    if hashseed_child:return summary
    import os
    seeds=[]
    for seed in ('0','1','42','20261027'):
        env=dict(os.environ,PYTHONHASHSEED=seed)
        result=subprocess.check_output([sys.executable,str(__file__),'--dir',str(dest),'--hashseed-child'],env=env)
        assert json.loads(result)==summary
        seeds.append(seed)
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    result=unittest.TextTestRunner(verbosity=2,stream=sys.stderr).run(suite)
    assert result.wasSuccessful()
    summary.update(status='PASS_LOCAL_PENDING_PROVIDER_AND_DUPLICATE_RUN_GATE',synthetic_tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),pythonhashseeds=seeds,
        no_model_calls=True,no_M1_gold_metrics_computed=True,provider_budget_verified=False)
    save(dest/'m1-preflight.json',summary)
    return {k:v for k,v in summary.items() if k!='per_document'}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--dir',type=Path,required=True);ap.add_argument('--hashseed-child',action='store_true')
    a=ap.parse_args();print(json_text(audit(a.dir,a.hashseed_child)))
