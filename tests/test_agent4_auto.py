"""Synthetic tests of the explicitly user-authorized automatic-only V0."""
from copy import deepcopy
import unittest
from tests.test_agent4_v0 import fixture
from agent4_v0.automated_run import structural, audit_passes, compile_tasks, helper_body, native_limits, require_never_submitted
from agent4_v0.common import save
from pathlib import Path
import tempfile
from agent4_v0.adapter import payload


class AutomaticV0Tests(unittest.TestCase):
    def proposal(self):
        return {'status':'proposed','operation':'neutralize','support_quotes':['Fire made smoke.'],
                'alternative_support_quotes':[], 'variants':{name:[{'before':word,'after':'preceded','occurrence':0}]
                for name,word in zip(('target','placebo_1','placebo_2','placebo_3'),('made','delayed','broke','stopped'))}}

    def test_structure_preserves_every_mention(self):
        d,_,_=fixture();variants=structural(d,self.proposal())
        for v in variants.values():
            self.assertTrue(all(v['text'][m['start']:m['end']]==m['surface'] for m in v['mentions']))

    def test_protected_event_edit_rejected(self):
        d,_,_=fixture();p=self.proposal();p['variants']['target'][0]['before']='Fire'
        with self.assertRaises(ValueError):structural(d,p)

    def test_invented_quote_rejected(self):
        d,_,_=fixture();p=self.proposal();p['support_quotes']=['not in the original']
        with self.assertRaises(ValueError):structural(d,p)

    def test_decline_not_forced_into_task(self):
        d,_,_=fixture()
        with self.assertRaises(ValueError):structural(d,{'status':'declined'})

    def test_blind_auditor_all_four_required(self):
        mapping={'A':'target','B':'placebo_1','C':'placebo_2','D':'placebo_3'}
        verdict={'original_support':'documented','effects':[{'variant_id':letter,
            'effect':'weaken_pair_support' if letter=='A' else 'preserve_pair_support',
            'grammar_ok':True,'facts_added':False,'incident_identity_preserved':True,'modality_preserved':True}
            for letter in 'ABCD']}
        self.assertEqual(len(audit_passes(verdict,mapping)),4)
        verdict['effects'][1]['effect']='unclear'
        with self.assertRaises(ValueError):audit_passes(verdict,mapping)

    def test_no_fake_human_review_and_seven_reads(self):
        d,_,s=fixture();v=structural(d,self.proposal())
        tasks=compile_tasks({'doc_id':d['doc_id'],'pair':['1','2'],'topic_id':'1'},d,v)
        self.assertEqual(len(tasks),7)
        self.assertEqual(sum(t['variant']=='original' for t in tasks),2)
        self.assertTrue(all(t['validation_kind']=='AUTOMATIC_PROVISIONAL_NOT_HUMAN_REVIEWED' for t in tasks))
        self.assertTrue(all('reviewed_blind_to_gold' not in t for t in tasks))
        originals=[payload(t,s,d) for t in tasks if t['variant']=='original']
        self.assertEqual(originals,[s['request'],s['request']])

    def test_native_uniform_slot_budget_covers_native_guard(self):
        costs=[.008,.012,.009]
        slot,total=native_limits(costs)
        self.assertGreaterEqual(total,len(costs)*slot)
        self.assertGreater(total,sum(costs))

    def test_only_proven_pre_post_stages_can_reuse_reservation(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td);jobs=[{'id':'synthetic','body':{}}]
            save(path/'request-plan.json',{'jobs':jobs})
            require_never_submitted(path,jobs)
            (path/'luna-batch-jobs').mkdir()
            with self.assertRaises(ValueError):require_never_submitted(path,jobs)

    def test_helper_has_no_demonstration_or_extractor_prompt(self):
        body=helper_body('editor',{'text':'synthetic'},16384)
        self.assertEqual(len(body['messages']),2)
        self.assertEqual(body['max_tokens'],16384)
        self.assertEqual(body['model'],'openai/gpt-6-luna:batch')


if __name__=='__main__':unittest.main()
