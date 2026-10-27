import copy
import unittest
from eci_alias.core import (Document, prepare_view, choose_mentions, make_p_renaming,
    validate_pair_view, project_one_generation, aggregate, unordered)
from eci_alias.eval import measures, evaluate, parse_pair_list


def dct():
    return {
        "doc_id":"synthetic001", "topic":"X",
        "text":"Une <e1 explosion> provoque un <e2 incendie>. Plus tard, <e3 explosion> ailleurs.",
        "tokens":["Une","explosion","provoque","un","incendie",".","Plus","tard",",","explosion","ailleurs","."],
        "mentions":["e1","e2","e3"],
        "spans":{"e1":[1],"e2":[4],"e3":[9]},
        "candidates":[["e1","e2"],["e1","e3"],["e2","e3"]],
    }

class TestPublicInput(unittest.TestCase):
    def setUp(self): self.raw=dct();self.doc=Document.from_public_json(dct())

    def test_expected_doc(self): self.assertEqual(len(self.doc.candidates),3)
    def test_gold_fields_rejected(self):
        for key in ("gold","annots","relations","positive_pairs","labels"):
            raw=dct();raw[key]=[]
            with self.assertRaises(ValueError):Document.from_public_json(raw)
    def test_requires_explicit_candidates(self):
        raw=dct();raw.pop('candidates')
        with self.assertRaises(ValueError):Document.from_public_json(raw)
    def test_duplicated_mentions_rejected(self):
        raw=dct();raw["mentions"][1]="e1"
        with self.assertRaises(ValueError):Document.from_public_json(raw)
    def test_discontinuous_span_kept(self):
        raw=dct();raw['spans']['e1']=[1,3]
        doc=Document.from_public_json(raw)
        a=prepare_view(doc,'A',selected=['e1'])
        self.assertEqual(a.auxiliary_records[0]["source_token_indices"],[1,3])
        self.assertEqual(a.auxiliary_records[0]["quoted_tokens"],["explosion","un"])
    def test_invalid_span_rejected(self):
        raw=dct();raw['spans']['e1']=[1,13]
        with self.assertRaises(ValueError):Document.from_public_json(raw)
    def test_duplicate_span_index_rejected(self):
        raw=dct();raw['spans']['e1']=[1,1]
        with self.assertRaises(ValueError):Document.from_public_json(raw)
    def test_self_candidate_rejected(self):
        raw=dct();raw['candidates'].append(['e2','e2'])
        with self.assertRaises(ValueError):Document.from_public_json(raw)
    def test_candidate_order_dedup(self):
        raw=dct();raw['candidates'].append(['e2','e1'])
        self.assertEqual(len(Document.from_public_json(raw).candidates),3)
    def test_selection_deterministic(self):
        self.assertEqual(choose_mentions(self.doc),choose_mentions(self.doc))
        self.assertEqual(len(choose_mentions(self.doc)),1)
    def test_no_change_source_AR(self):
        a=prepare_view(self.doc,'A',selected=['e1','e2'])
        r=prepare_view(self.doc,'R',selected=['e1','e2'])
        report=validate_pair_view(self.doc,a,r)
        self.assertEqual(report['selected'],2)
        self.assertEqual(a.document_text,self.doc.text)
        self.assertEqual(r.document_text,self.doc.text)
    def test_alias_reflects_only_same_mention(self):
        a=prepare_view(self.doc,'A',selected=['e1','e2'])
        self.assertEqual(a.projection['v__0000'],'e1')
        self.assertEqual(a.projection['v__0001'],'e2')
        self.assertEqual(a.projection['e3'],'e3')
    def test_R_uses_no_new_ids(self):
        v=prepare_view(self.doc,'R',selected=['e1'])
        self.assertEqual(v.auxiliary_records[0]['handle_id'],'e1')
        self.assertEqual(set(v.projection),set(self.doc.mentions))
    def test_P_renames_only_annotation_tags(self):
        v=prepare_view(self.doc,'P')
        self.assertIn('Une ',v.document_text)
        self.assertIn(' explosion>',v.document_text)
        self.assertNotIn('<e1 ',v.document_text)
        self.assertEqual(set(v.projection.values()),set(self.doc.mentions))
    def test_P_requires_annotation_tags(self):
        raw=dct();raw['text']='explosion provoque incendie'
        with self.assertRaisesRegex(ValueError,'requires'):
            prepare_view(Document.from_public_json(raw),'P')
    def test_pair_direction_ignored(self):
        self.assertEqual(unordered('e1','e2'),unordered('e2','e1'))

class TestProjector(unittest.TestCase):
    def setUp(self):
        self.doc=Document.from_public_json(dct())
        self.view=prepare_view(self.doc,'A',selected=['e1','e2'])

    def test_collapse_many_edges_and_reverse(self):
        raw=[['v__0000','causes','v__0001'],['e1','causes','e2'],
             ['v__0001','causedby','e1'],['v__0000','causes','v__0000'],
             ['v__0000','causes','e1'],['bad','causes','e1']]
        pred,c=project_one_generation(raw,self.view)
        self.assertEqual(pred,{('e1','e2')})
        self.assertEqual(c['duplicate_after_projection'],2)
        self.assertEqual(c['self_loop'],2)
        self.assertEqual(c['unknown_id'],1)
    def test_unsupported_labels_do_not_vote(self):
        p,c=project_one_generation([['e1','no','e2'],['e1','uncertain','e2']],self.view)
        self.assertEqual(p,set())
        self.assertEqual(c['bad_label'],2)
    def test_unknown_id_rejected(self):
        p,c=project_one_generation([['e4','causes','e1']],self.view)
        self.assertEqual(p,set());self.assertEqual(c['unknown_id'],1)
    def test_noncandidate_rejected(self):
        raw=dct();raw['candidates']=[['e1','e2']]
        v=prepare_view(Document.from_public_json(raw),'B')
        p,c=project_one_generation([['e1','causes','e3']],v)
        self.assertEqual(p,set());self.assertEqual(c['not_candidate'],1)
    def test_malformed_triples_rejected(self):
        p,c=project_one_generation([[],['e1','e2'],['e1',True,'e2']],self.view)
        self.assertEqual(p,set());self.assertEqual(c['bad_format'],3)
    def test_seven_slots_four_needed(self):
        raw=[['v__0000','causes','e2']]
        responses=[raw,None,raw,[],raw,[],raw]
        result=aggregate(responses,self.view)
        self.assertEqual(result['predicted_pairs'],[['e1','e2']])
        self.assertEqual(result['vote_counts'][0]['count'],4)
        self.assertEqual(result['counts']['failed_or_empty_slot'],1)
    def test_seven_slots_three_fails(self):
        raw=[['e1','causes','e2']]
        responses=[raw,None,raw,[],raw,[],[]]
        self.assertEqual(aggregate(responses,self.view)['predicted_pairs'],[])
    def test_incomplete_passes_fail(self):
        with self.assertRaises(ValueError):aggregate([[]]*6,self.view)
    def test_json_wrapper_accepted(self):
        p,_=project_one_generation({'relations':[['e1','causes','e2']]},self.view)
        self.assertEqual(p,{('e1','e2')})

class TestEvaluation(unittest.TestCase):
    def setUp(self):
        self.doc=Document.from_public_json(dct())
        self.c={self.doc.doc_id:set(self.doc.candidates)}
        self.t={self.doc.doc_id:'X'}
        self.gold=[{"doc_id":self.doc.doc_id,"positive_pairs":[['e1','e2']]}]
    def test_f1_reference_math(self):
        self.assertAlmostEqual(measures(464,937,971)['F1'],928/1908)
    def test_delta_with_addition(self):
        pred=[{'doc_id':self.doc.doc_id,'arm':'A','predicted_pairs':[['e1','e2']]},
              {'doc_id':self.doc.doc_id,'arm':'R','predicted_pairs':[]}]
        report=evaluate(self.gold,pred,self.c,self.t)
        c=report['comparisons']['A_vs_R']
        self.assertEqual(c['added_TP'],1)
        self.assertEqual(c['removed_total'],0)
        self.assertGreater(c['utility'],0)
    def test_delta_with_removed_TP(self):
        pred=[{'doc_id':self.doc.doc_id,'arm':'A','predicted_pairs':[]},
              {'doc_id':self.doc.doc_id,'arm':'R','predicted_pairs':[['e1','e2']]}]
        report=evaluate(self.gold,pred,self.c,self.t)
        self.assertEqual(report['comparisons']['A_vs_R']['removed_TP'],1)
    def test_ci_under_three_topics_unavailable(self):
        pred=[{'doc_id':self.doc.doc_id,'arm':'A','predicted_pairs':[]},
              {'doc_id':self.doc.doc_id,'arm':'R','predicted_pairs':[]}]
        report=evaluate(self.gold,pred,self.c,self.t)
        self.assertIsNone(report['comparisons']['A_vs_R']['ci']['ci95'])
    def test_gold_outside_candidates_fails(self):
        gold=[{'doc_id':self.doc.doc_id,'positive_pairs':[['e1','unknown']]}]
        pred=[{'doc_id':self.doc.doc_id,'arm':'A','predicted_pairs':[]}]
        with self.assertRaises(ValueError):evaluate(gold,pred,self.c,self.t)
    def test_arm_without_all_docs_fails(self):
        gold=[*self.gold,{'doc_id':'doc2','positive_pairs':[]}]
        c={**self.c,'doc2':set()}
        pred=[{'doc_id':self.doc.doc_id,'arm':'A','predicted_pairs':[]}]
        with self.assertRaises(ValueError):evaluate(gold,pred,c,{**self.t,'doc2':'Y'})

if __name__ == '__main__':unittest.main()

class TestBudgetGate(unittest.TestCase):
    def _usage(self):
        rows=[]
        for p in range(7):
            for arm in ('A','R'):
                rows.append({'doc_id':'doc1','pass_index':p,'arm':arm,
                    'input_tokens':1000+(1 if arm=='A' else 0),
                    'completion_tokens':100,
                    'model_version':'frozen','demo_id':str(p),
                    'base_prompt_sha256':'unchanged','decode_sha256':'unchanged',
                    'truncated':False})
        return rows
    def test_same_budget_passes(self):
        from eci_alias.budget import verify
        self.assertEqual(verify(self._usage())['status'],'PASS')
    def test_large_delta_blocked(self):
        from eci_alias.budget import verify
        rows=self._usage();rows[0]['input_tokens']=1200
        self.assertEqual(verify(rows)['status'],'BLOCKED')
    def test_missing_pass_blocked(self):
        from eci_alias.budget import verify
        self.assertEqual(verify(self._usage()[:-1])['status'],'BLOCKED')
    def test_different_demo_blocked(self):
        from eci_alias.budget import verify
        rows=self._usage();rows[0]['demo_id']='different'
        self.assertEqual(verify(rows)['status'],'BLOCKED')
    def test_truncation_blocked(self):
        from eci_alias.budget import verify
        rows=self._usage();rows[0]['truncated']=True
        self.assertEqual(verify(rows)['status'],'BLOCKED')
    def test_missing_provider_usage_blocked(self):
        from eci_alias.budget import verify
        rows=self._usage();rows[0].pop('input_tokens')
        self.assertEqual(verify(rows)['status'],'BLOCKED')
