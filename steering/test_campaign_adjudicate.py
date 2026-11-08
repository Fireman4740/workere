"""C3 inference-free checks for routing, order, fallback and complete scoring."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from steering import campaign_adjudicate as a


class AdjudicationTests(unittest.TestCase):
    def test_route_uses_valid_disagreements_only(self):
        p, q, r = ("1","2"), ("1","3"), ("2","3")
        self.assertEqual(a.route_pairs({p,q,r}, {p}, [{q:True}, {p:True,r:False}]), {q})
        self.assertEqual(a.route_pairs({p,q,r}, {p}, [{p:False}, {}]), {p})

    def test_parse_requires_entire_final_json(self):
        self.assertEqual(a.parse_final('{"label":"A"}', 'direct'), {'label':'A'})
        for text in ('{"label":"A","label":"N"}', '<think>A</think>{"label":"A"}',
                     '{"label":"C"}', '{"label":"A","extra":1}', '{"label":"A"} extra'):
            self.assertIsNone(a.parse_final(text,'direct'))
        self.assertIsNone(a.parse_final('{"label":"A"}','evidence'))

    def test_grounding_checks_event_identity_and_link(self):
        text = 'The <1 blast> caused the <2 damage>.'
        obj = dict(label='A',event_a_quote='<1 blast>',event_b_quote='<2 damage>',
                   link_quote=text,alternative='Temporal sequence alone is insufficient.')
        self.assertTrue(a.grounded(obj,text,('1','2')))
        self.assertFalse(a.grounded(obj,text,('2','1')))
        self.assertFalse(a.grounded({**obj,'link_quote':'fabricated support'},text,('1','2')))
        self.assertFalse(a.grounded({**obj,'link_quote':''},text,('1','2')))
        self.assertTrue(a.grounded({**obj,'label':'N','link_quote':''},text,('1','2')))

    def test_swapped_presentations_require_consistent_direction(self):
        self.assertEqual(a.canonical_label('B',True),'A')
        self.assertEqual(a.canonical_label('N',True),'N')
        self.assertFalse(a.merge_decision(False,['A','B'],a.PRIMARY))
        self.assertTrue(a.merge_decision(False,['A','A'],a.PRIMARY))
        self.assertFalse(a.merge_decision(True,['N','N'],a.PRIMARY))
        self.assertTrue(a.merge_decision(True,['N','N'],'additions_only'))

    def test_invalid_judgment_keeps_baseline_not_negative(self):
        for baseline in (False,True):
            for labels in ([None,'N'],['A',None],[None,None],['A','N']):
                self.assertEqual(a.merge_decision(baseline,labels,a.PRIMARY),baseline)

    def test_prompt_hides_prior_votes_and_keeps_source(self):
        context = 'The <1 blast> caused <2 damage>.'
        for arm in a.ARMS:
            msgs = a.messages('Original annotation rules.',context,('2','1'),arm)
            self.assertIn(context,msgs[1]['content'])
            self.assertIn('A=<2>, B=<1>',msgs[1]['content'])
            self.assertNotIn('vote',json.dumps(msgs).lower())

    def test_end_to_end_prepare_and_score_all_documents(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, work, data, out = [root / name for name in ('C2','work','data','C3')]
            source.mkdir(); data.mkdir(); (data/'esc').mkdir()
            for name in ('ctb-suite.json','ctb-prompt.json','reference.json',
                         'esc/test-s40.json','esc/prompt.json','esc/baseline.json'):
                (data/name).write_text('{}')
            for name in ('N1','P3','P2'):
                (work/f'out/{name}').mkdir(parents=True)
                (work/f'out/{name}/gen.jsonl').write_text('')
            text = 'The <1 blast> caused <2 damage>. The <3 meeting> occurred later.'
            universe = [('1','2'),('2','1'),('1','3'),('3','1')]
            ctb = {
                'd': dict(id='d',text=text,pair_list=universe,gold=[('1','causes','2'),('2','causedby','1')]),
                'empty': dict(id='empty',text=text,pair_list=universe,gold=[]),
            }
            esc = {'1_doc':dict(id='1_doc',text=text,pair_list=universe,gold=['1,2','2,1'])}
            selected = {'CTB':['d','empty'],'ESC':['1_doc']}
            docs = {'CTB':ctb,'ESC':esc}
            baseline_predictions = {
                'CTB':{name:{'d':{('1','3'),('3','1')},'empty':set()} for name in ('qwen7','n1_p3')},
                'ESC':{name:{'1_doc':{('1','3'),('3','1')}} for name in ('qwen_p2_sym','luna7','fusion')},
            }
            baselines = {}
            for dataset, names in baseline_predictions.items():
                baselines[dataset] = {}
                for name,predictions in names.items():
                    baselines[dataset][name] = {}
                    for doc_id,pred in predictions.items():
                        gold, candidates = a.f.gold_and_universe(dataset,docs[dataset][doc_id])
                        baselines[dataset][name][doc_id] = a.c.counts(pred,gold,candidates)
            a.c.dump(source/'manifest.json',dict(selected=selected))
            a.c.dump(source/'model_identity.json',dict(model='fixture'))
            a.c.dump(source/'report.json',dict(selected=selected,baselines={
                ds:{n:a.c.metric(per) for n,per in names.items()} for ds,names in baselines.items()}))
            requests,rows = [],[]
            for dataset,ids in selected.items():
                for doc_id in ids:
                    for schedule in (0,1):
                        req = dict(key=f'{dataset}:{doc_id}:{schedule}',condition='plain_4096',
                                   dataset=dataset,doc_id=doc_id,schedule=schedule,pairs=[('1','2'),('1','3')])
                        requests.append(req)
                        rows.append(dict(key=req['key'],status='valid',answer=json.dumps({
                            '0':'A' if doc_id!='empty' else 'N','1':'N'})))
            a.write_jsonl(source/'requests.jsonl',requests)
            a.write_jsonl(source/'generations.jsonl',rows)
            with patch.object(a.c,'DATA',data), patch.object(a.c,'inputs',return_value=(ctb,{},esc,{'CTB':'Rules','ESC':'Rules'})), \
                 patch.object(a.f,'historical_baselines',return_value=(baselines,baseline_predictions)), redirect_stdout(io.StringIO()):
                a.prepare(out,source,work)
                reqs = a.c.jsonl(out/'requests.jsonl')
                self.assertEqual(len(reqs),16)
                self.assertTrue(all(r['doc_id']!='empty' for r in reqs))
                self.assertTrue(all('gold' not in r and 'baseline' not in r for r in reqs))
                results = []
                for req in reqs:
                    positive = req['pair']==['1','2']
                    obj = dict(label=('B' if req['reverse'] else 'A') if positive else 'N')
                    if req['arm']=='evidence':
                        spans = {m.group(1):m.group(0) for m in a.c.TAG.finditer(req['context'])}
                        obj.update(event_a_quote=spans[req['presented'][0]],event_b_quote=spans[req['presented'][1]],
                                   link_quote=req['context'] if positive else '',alternative='Only a later event for the unrelated pair.')
                    results.append(dict(key=req['key'],status='valid',answer=json.dumps(obj),thinking_forced=False,
                        prompt_tokens=10,total_input_tokens=30,think_tokens=10,answer_tokens=5,wall_seconds_share=.1))
                a.write_jsonl(out/'generations.jsonl',results)
                a.score(out)
                report = json.loads((out/'report.json').read_text())
                self.assertEqual(report['conditions']['CTB/evidence/consensus_repair']['f1'],100)
                self.assertEqual(report['conditions']['CTB/evidence/consensus_repair']['documents'],2)
                self.assertEqual(report['conditions']['ESC/direct/consensus_repair']['f1'],100)
                self.assertEqual(report['contrasts']['CTB/evidence_vs_direct/consensus_repair'],[0,0,0])
                self.assertEqual(report['conditions']['CTB/evidence/consensus_repair']['changes_unordered']['removed_fp_links'],1)
                self.assertEqual(report['conditions']['CTB/evidence/additions_only']['fp'],2)
                manifest = json.loads((out/'manifest.json').read_text())
                self.assertTrue(all(not k.startswith(str(root)) for k in manifest['source_hashes']))
                # Re-preparing may not replace the frozen requests with a new routing scheme.
                a.prepare(out,source,work)


if __name__ == '__main__':
    unittest.main()
