import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from steering import maven_recall as mr
from steering import maven_budget as mb
from steering.test_maven_pool import DOC


TEXT='Document:\n<e1 attack> by Alice occurred. <e2 fled> Alice then. <e3 explosion> followed.'
STRUCTURE=dict(sentence_of=dict(e1=0,e2=1,e3=5))


def base(p=0):
    return dict(key=str(p),doc_id='d',pass_id=p,demo_id='train'+str(p),messages=[
        dict(role='system',content='old rules'),dict(role='user',content='TRAIN DOCUMENT'),
        dict(role='assistant',content=json.dumps(dict(relations=[dict(pair='e1,e2',label='causes')]))),
        dict(role='user',content=TEXT)])


class RecallTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg=mr.config('steering/maven_recall_config.json')
        self.cfg.update(campaign_root=self.tmp.name,bootstrap_draws=100)

    def test_expansion_is_cartesian_without_transitive_causality(self):
        graph=dict(clusters=[['e1','e2']],edges=[dict(cause='e1',effect='e3'),dict(cause='e3',effect='e4')])
        raw, expanded=mr.parse_graph(graph,['e1','e2','e3','e4'],'coref')
        self.assertEqual(raw,{('e1','e3'),('e3','e4')})
        self.assertEqual(expanded,{('e1','e3'),('e2','e3'),('e3','e4')})
        self.assertNotIn(('e1','e4'),expanded)

    def test_overlapping_unknown_self_edges_and_control_merges_are_invalid(self):
        for graph,mode in [
            (dict(clusters=[['e1','e2'],['e2','e3']],edges=[]),'coref'),
            (dict(clusters=[['unknown']],edges=[]),'coref'),
            (dict(clusters=[['e1','e2']],edges=[dict(cause='e1',effect='e2')]),'coref'),
            (dict(clusters=[['e1','e2']],edges=[]),'singleton'),
            (dict(clusters=[],edges=[dict(cause='e1',effect='other')]),'coref')]:
            with self.subTest(graph=graph,mode=mode),self.assertRaises(ValueError):
                mr.parse_graph(graph,['e1','e2','e3'],'singleton' if mode=='singleton' else 'coref')

    def test_omitted_mentions_become_singletons(self):
        groups=mr.groups_of([['e1','e2']],['e1','e2','e3'])
        self.assertEqual(groups['e3'],['e3'])
        self.assertEqual(mr.propagate({('e1','e3')},[['e1','e2']],groups),{('e1','e3'),('e2','e3')})

    def test_candidates_are_gold_blind_exclude_baseline_and_intra(self):
        s=dict(sentence_of=dict(e1=0,e2=0,e3=1,e4=3,e5=7))
        text=' '.join('<'+e+' event Alice>' for e in s['sentence_of'])
        universe=set(__import__('itertools').combinations(sorted(s['sentence_of']),2))
        baseline={('e1','e3')}
        got=mr.candidates('d',text,s,universe,baseline,6,42)
        self.assertNotIn(('e1','e2'),got)
        self.assertNotIn(('e1','e3'),got)
        self.assertEqual(len(got),6)
        self.assertEqual(got,mr.candidates('d',text,s,universe,baseline,6,42))
        distances={abs(s['sentence_of'][x]-s['sentence_of'][y]) for x,y in got}
        self.assertIn(1,distances)
        self.assertTrue(any(2<=d<=3 for d in distances))
        self.assertTrue(any(d>=4 for d in distances))

    def test_public_jobs_unchanged_when_gold_changes(self):
        for kind in ['coref','search']:
            cfg=dict(self.cfg,kind=kind)
            docs={'d':DOC}
            first,candidates=mr.build_jobs(cfg,[base(i) for i in range(5)],docs,{'d':STRUCTURE},{'d':set()})
            other=copy.deepcopy(docs);other['d']['gold']=['SECRET']
            second,_=mr.build_jobs(cfg,[base(i) for i in range(5)],other,{'d':STRUCTURE},{'d':set()})
            self.assertEqual(first,second)
            self.assertNotIn('SECRET',json.dumps(first))
            self.assertEqual(len(first),10)
            for i in range(0,len(first),2):
                self.assertEqual(first[i]['pairs'],first[i+1]['pairs'])

    def test_grounding_rejects_invented_quote_argument_or_event(self):
        good=dict(id='e1',participant='Alice',quote='<e1 attack> by Alice')
        self.assertEqual(mr.grounded_frames(dict(frames=[good]),TEXT,{'e1'}),[good])
        for patch_ in [dict(quote='invented Alice'),dict(participant='Bob'),dict(id='e2'),dict(id=[])]:
            self.assertIsNone(mr.grounded_frames(dict(frames=[dict(good,**patch_)]),TEXT,{'e1','e2'}))

    def reader(self,kind):
        cfg=dict(self.cfg,kind=kind)
        jobs,_=mr.build_jobs(cfg,[base()],{'d':DOC},{'d':STRUCTURE},{'d':set()})
        r=object.__new__(mr.Reader)
        r.cfg,r.out,r.manifest=cfg,mb.output_dir(cfg),{'configuration':cfg,'engine_sha256':'engine'}
        return r,jobs

    @patch.dict(os.environ,ORX_RUN_ID='test',ORX_COMMIT='commit')
    def test_exactly_matched_generation_caps_and_identical_candidates(self):
        r,jobs=self.reader('search')
        budgets,seen=[],[]
        def stage(job,name,messages,schema,thinking,answer):
            budgets.append(thinking+answer);seen.append(messages[-1]['content'])
            parsed=dict(frames=[]) if name=='participants' else dict(labels={str(i):'N' for i in range(len(job['pairs']))})
            return dict(parsed=parsed,think_tokens=1,answer_tokens=1,prompt_tokens=1,final_input_tokens=1,
                        seconds=.1,thinking_forced=False,answer_finish='stop')
        r.stage=stage
        totals={}
        for j in jobs:
            budgets.clear()
            row,hit=r.one(j)
            self.assertTrue(row['valid']);self.assertFalse(hit)
            totals[j['mode']]=sum(budgets)
            self.assertEqual(r.one(j),(row,True))
        self.assertEqual(totals,dict(direct=6656,structured=6656))
        self.assertEqual(jobs[0]['pairs'],jobs[1]['pairs'])

    @patch.dict(os.environ,ORX_RUN_ID='test',ORX_COMMIT='commit')
    def test_invalid_graph_is_an_empty_read_not_regenerated(self):
        r,jobs=self.reader('coref')
        def bad(*args):
            return dict(parsed=None,think_tokens=1,answer_tokens=1,prompt_tokens=1,final_input_tokens=1,
                        seconds=.1,thinking_forced=True,answer_finish='length')
        r.stage=bad
        row,hit=r.one(jobs[0])
        self.assertFalse(row['valid']);self.assertFalse(row['pairs'])
        self.assertEqual(r.one(jobs[0]),(row,True))

    def test_historical_incomplete_panel_refused(self):
        rows=[dict(doc='d',budget='long',status='valid',pairs=['e1,e2'],**{'pass':p}) for p in range(5)]
        self.assertEqual(mr.historical_views(dict(reads=rows),{'d':DOC}),{'d':{('e1','e2')}})
        with self.assertRaises(ValueError):mr.historical_views(dict(reads=rows[:-1]),{'d':DOC})
        with self.assertRaises(ValueError):mr.historical_views(dict(reads=rows+rows[:1]),{'d':DOC})

    def test_missing_outputs_prohibit_score(self):
        r,jobs=self.reader('coref')
        manifest=dict(configuration=r.cfg,jobs_sha256=mr.api.digest(jobs))
        for name,value in [('manifest',manifest),('jobs',jobs),('documents',{'d':DOC}),('doc_structure',{'d':STRUCTURE})]:
            mb.atomic(r.out/(name+'.json'),value)
        with self.assertRaises(FileNotFoundError):mr.score(r.cfg)
        self.assertFalse((r.out/'report.json').exists())

    def test_invalid_votes_keep_fixed_denominator_and_changes_count_fp(self):
        readings=[{('e1','e2')}, {('e1','e2')},set(),set(),set()]
        self.assertFalse(mp_vote:=mr.mp.vote(readings,3))
        count=mr.counts_for({'d':mp_vote},{'d':DOC},{'d':STRUCTURE})
        self.assertEqual(count['all']['d'],dict(tp=0,fp=0,fn=1))
        self.assertEqual(mr.changes({'d':{('e1','e2'),('e1','e3')}},{'d':set()},{'d':DOC})['added_fp'],1)

    @patch.dict(os.environ,ORX_RUN_ID='test',ORX_COMMIT='commit')
    def test_complete_coref_panel_scores_paired_metrics_and_oracle_separately(self):
        cfg=dict(self.cfg,kind='coref')
        jobs,_=mr.build_jobs(cfg,[base(i) for i in range(5)],{'d':DOC},{'d':STRUCTURE},{'d':set()})
        out=mb.output_dir(cfg)
        manifest=dict(configuration=cfg,jobs_sha256=mr.api.digest(jobs))
        for name,value in [('manifest',manifest),('jobs',jobs),('documents',{'d':DOC}),
                           ('doc_structure',{'d':STRUCTURE}),('baseline',{'d':[]}),
                           ('oracle_diagnostic_only',{'d':[]})]:
            mb.atomic(out/(name+'.json'),value)
        for job in jobs:
            pairs=[['e1','e2']] if job['mode']=='coref' and job['pass_id']<3 else []
            data=dict(key=job['key'],doc_id='d',pass_id=job['pass_id'],mode=job['mode'],
                      valid=job['pass_id']<3,evidence_invalid=False,pairs=pairs,raw=pairs,
                      cost=dict(think_tokens=1,answer_tokens=1),forced=0,truncated=0,run='test')
            mb.atomic(out/'responses'/(job['key']+'.json'),dict(contract=mr.api.digest([manifest,job]),
                      sha256=mr.api.digest(data),data=data))
        with patch('builtins.print'):
            report=mr.score(cfg)
        primary=report['contrasts']['coref_ge3 - singleton_ge3']
        self.assertEqual(primary['delta_ci95'],(100.0,100.0,100.0))
        self.assertTrue(primary['primary'])
        self.assertIn('oracle_propagation_diagnostic',report['metrics'])
        self.assertFalse(any('oracle' in c for c in report['contrasts']))
        self.assertTrue((out/'complete.json').is_file())


if __name__=='__main__':unittest.main()
