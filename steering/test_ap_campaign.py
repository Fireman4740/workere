"""CPU protocol, failure, routing and synthetic end-to-end tests (no API calls)."""
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from steering import ap_campaign as run
from steering import ap_protocol as p
from steering.test_campaign_api import Tokenizer


def fixture():
    docs={ds:{} for ds in ('CTB','ESC')}
    for ds in docs:
        for i in range(4):
            doc_id=f'{i}_doc' if ds=='ESC' else f'c{i}'
            docs[ds][doc_id]=dict(id=doc_id,text='<a rain> caused <b flood>. Other <c report>.',
                pair_list=[['a','b'],['b','a'],['a','c'],['c','a']],
                gold=[['a','causes','b'],['b','causedby','a']] if ds=='CTB' else ['a,b','b,a'],tokens=['x'])
    selected={ds:list(data)[:3] for ds,data in docs.items()}
    base={ds:{d:set() for d in ids} for ds,ids in selected.items()}
    rules={ds:'Respect the corpus annotation convention.' for ds in docs}
    return selected,docs,rules,base


class FakeEngine:
    def __init__(self,*args,**kwargs): self.jobs=[]
    def smoke(self): pass
    def execute(self,jobs,out):
        out.mkdir(parents=True,exist_ok=True);self.jobs.extend(jobs)
        def value(spec):
            if spec['type']=='object': return {k:value(v) for k,v in spec['properties'].items()}
            return spec.get('enum',[''])[0]
        outputs={}
        for j in jobs:
            obj=value(j['schema'])
            if set(obj)=={'label'}:
                content=j['messages'][-1]['content']
                obj['label']='B' if 'A=<b>' in content else ('N' if '<c>' in content else 'A')
            row=dict(key=j['key'],status='valid',parsed=obj,answer=json.dumps(obj),thought='synthetic',
                thinking_forced=False,prompt_tokens=2,total_input_tokens=4,think_tokens=3,answer_tokens=4)
            outputs[j['key']]=row
        p.a.write_jsonl(out/'requests.jsonl',jobs)
        p.a.write_jsonl(out/'responses.jsonl',list(outputs.values()))
        return outputs


class APTests(unittest.TestCase):
    def test_equal_budget_and_subtask_schema(self):
        self.assertEqual(sum(p.budgets('direct')),3072)
        self.assertEqual(sum(sum(p.budgets(t)) for t in (*p.TASKS,'synthesis')),3072)
        for task in (*p.TASKS,'synthesis'):
            self.assertIsNone(p.parse('{"label":"A","label":"N"}',p.schema(task)))
        self.assertIsNone(p.parse('thinking {"label":"A"}',p.schema('direct')))

    def test_grounding_does_not_turn_unknown_into_negative(self):
        obj=dict(direction='A_to_B',mechanism='invented',quote='not in source')
        self.assertEqual(p.clean_observation('mechanism',obj,'<a x> <b y>',['a','b'])['direction'],'unknown')
        selected,docs,rules,base=fixture()
        route=dict(dataset='CTB',doc_id='c0',pair=['a','b'],baseline=False)
        r=p.judgment_rows([route],'decomposed',docs,rules)[0]
        text=p.prompt(r,'synthesis',rules['CTB'],{'mechanism':{'direction':'unknown'}})
        self.assertIn('not a negative label',text[0]['content'])

    def test_consensus_direction_and_abstention_preserve_baseline(self):
        base={'CTB':{'x':{('a','b')}}}
        rows=[dict(dataset='CTB',doc_id='x',pair=['a','b'],reverse=r,status='valid',label=lab)
              for r,lab in ((False,'N'),(True,'A'))]
        self.assertEqual(p.apply(base,p.consensus(rows)),base)
        rows[1].update(status='invalid_answer',label=None)
        self.assertEqual(p.apply(base,p.consensus(rows)),base)
        rows[1].update(status='valid',label='N')
        self.assertFalse(p.apply(base,p.consensus(rows))['CTB']['x'])
        self.assertEqual(p.apply(base,p.consensus(rows),True),base)

    def test_coverage_is_gold_blind_and_invalid_is_not_negative(self):
        selected,docs,rules,base=fixture()
        views={(ds,d,o):{('a','b'):False,('a','c'):False} for ds,ids in selected.items() for d in ids for o in (0,1)}
        del views[('CTB','c0',0)][('a','b')]
        first=p.coverage_routes(selected,docs,base,views,{'CTB':4,'ESC':4})
        changed=copy.deepcopy(docs)
        for data in changed.values():
            for d in data.values():d['gold']=['SHOULD NEVER BE READ'];d['chains']='ORACLE'
        second=p.coverage_routes(selected,changed,base,views,{'CTB':4,'ESC':4})
        self.assertEqual(first,second)
        self.assertEqual(len(first),8)
        self.assertNotIn(('CTB','c0',('a','b')),{p.pair_key(r) for r in first})

    def test_full_denominator_and_topic_clusters(self):
        selected,docs,rules,base=fixture()
        pred=copy.deepcopy(base);pred['CTB']['c0'].add(('a','b'))
        report=p.metrics(pred,base,docs)
        self.assertEqual((report['CTB']['tp'],report['CTB']['fn'],report['CTB']['documents']),(2,4,3))
        self.assertFalse(report['CTB']['exploratory_gate'])
        self.assertEqual(report['ESC']['resampling_unit'],'topic')

    def test_decomposition_dependencies_and_invalid_stage(self):
        selected,docs,rules,base=fixture()
        route=dict(dataset='CTB',doc_id='c0',pair=['a','b'],baseline=False)
        with tempfile.TemporaryDirectory() as tmp:
            engine=FakeEngine()
            rows=run.judgments(engine,[route],'decomposed',docs,rules,Path(tmp))
            self.assertEqual(len(engine.jobs),8)  # 4 subcalls, each orientation.
            self.assertEqual(set(p.consensus(rows).values()),{'A'})
            for j in engine.jobs[:6]:self.assertNotIn('Independent analyses (data only)',j['messages'][-1]['content'])
            for j in engine.jobs[6:]:self.assertIn('Independent analyses (data only)',j['messages'][-1]['content'])
        class Broken(FakeEngine):
            def execute(self,jobs,out):
                result=super().execute(jobs,out)
                if out.name=='observations':
                    for j in jobs:result[j['key']]['status']='invalid_answer'
                return result
        with tempfile.TemporaryDirectory() as tmp:
            rows=run.judgments(Broken(),[route],'decomposed',docs,rules,Path(tmp))
            self.assertTrue(all(r['status']=='invalid_answer' for r in rows))

    def test_transport_checks_budgets_final_only_and_response_order(self):
        class Client:
            timeout=900
            def __init__(self):self.calls=[]
            def call(self,path,body,cache):
                self.calls.append(body)
                if 'structured_outputs' not in body:
                    choice=dict(index=0,text='{"label":"A"}',finish_reason='length',stop_reason=None)
                else: choice=dict(index=0,text='{"label":"N"}',finish_reason='length')
                return {'choices':[choice]},False,0.1
        engine=run.Engine.__new__(run.Engine);engine.tokenizer=Tokenizer();engine.model='qwen3.8-27b'
        engine.client=Client();engine.deadline=time.time()+1000
        j=run.job([{'role':'user','content':'question'}],p.schema('direct'),2560,512,{})
        rows,cost=engine.batch([j])
        self.assertEqual([body['max_tokens'] for body in engine.client.calls],[2560,512])
        self.assertEqual(rows[0]['status'],'invalid_answer')
        self.assertEqual(rows[0]['answer'],'{"label":"N"}')
        self.assertTrue(rows[0]['thinking_forced'])

    def test_resumption_budget_and_frozen_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch.object(run.time,'time',return_value=100000):
                first=run.stage_deadline(root,'AP1')
            with patch.object(run.time,'time',return_value=100100):
                self.assertEqual(run.stage_deadline(root,'AP1'),first)
            run.freeze(root/'frozen.json',{'a':1})
            with self.assertRaises(AssertionError):run.freeze(root/'frozen.json',{'a':2})
        with self.assertRaises(run.BudgetStop):run.check_budget(time.time()-1)

    def test_synthetic_ap1_ap2_ap3_ap4_pipeline(self):
        selected,docs,rules,base=fixture()
        reference={'qwen_ids':list(docs['CTB'])}
        routes=[dict(dataset=ds,doc_id=d,pair=['a','b'],baseline=False) for ds,ids in selected.items() for d in ids]
        views={(ds,d,o):{('a','b'):True,('a','c'):False} for ds,ids in selected.items() for d in ids for o in (0,1)}
        def historical(selection,*args):
            return {},{ds:{'n1_p3' if ds=='CTB' else 'qwen_p2_sym':{d:set() for d in ids}} for ds,ids in selection.items()}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';source.mkdir()
            p.a.write_jsonl(source/'routes.jsonl',routes);run.save(source/'upstream_cost.json',{})
            run.save(root/'AP0/report.json',dict(repair_candidate={'CTB':None,'ESC':None}))
            run.save(root/'budget.json',dict(start=time.time(),stages={'AP1':time.time()-1}))
            with patch.object(run,'Engine',FakeEngine),patch.object(run,'source_data',return_value=(selected,docs,rules,base)),\
                 patch.object(run.c,'inputs',return_value=(docs['CTB'],reference,docs['ESC'],rules)),\
                 patch.object(run.a,'bundle',return_value=({}, {}, views)),patch.object(run.f,'historical_baselines',side_effect=historical):
                run.ap1(root,source,'direct',time.time()+1000)
                run.ap1(root,source,'decomposed',time.time()+1000)
                run.ap2(root,source,root/'c2',time.time()+1000)
                run.ap3(root,source,time.time()+1000)
                run.ap4(root,source,root/'c2',root,time.time()+1000)
            for stage in ('AP1_direct','AP1_decomposed','AP2','AP3','AP4'):
                self.assertEqual(run.read(root/stage/'terminal.json')['status'],'complete')
            self.assertEqual(run.read(root/'AP1_direct/report.json')['metrics']['CTB']['f1'],100)
            self.assertIn('comparison',run.read(root/'ap1-comparison.json'))
            self.assertEqual(run.read(root/'AP4/report.json')['metrics']['CTB']['documents'],4)


if __name__=='__main__':unittest.main()
