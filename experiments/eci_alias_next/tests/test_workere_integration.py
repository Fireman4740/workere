"""Offline regression tests added during workere integration; no API calls."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from eci_alias.budget import verify
from eci_alias.core import Document, prepare_view, validate_pair_view
from eci_alias.eval import evaluate
from eci_alias.workere_run import prepare, parse_response, byte_bound


def usage():
    return [dict(doc_id='doc', arm=arm, pass_index=p, input_tokens=1000,
                 completion_tokens=100, model_version='frozen', demo_id=str(p),
                 base_prompt_sha256='a'*64, decode_sha256='b'*64, truncated=False,
                 demo_sha256='c'*64, full_prompt_sha256=('d' if arm=='A' else 'e')*64,
                 token_count_source='provider_usage', auxiliary_tokens_upper_bound=700,
                 served_model='frozen', status='ok')
            for p in range(7) for arm in ('A','R')]


def document():
    return Document.from_public_json(dict(doc_id='d', text='<a X> <b Y>', tokens=['X','Y'],
              mentions=['a','b'], spans={'a':[0],'b':[1]}, candidates=[['a','b']]))


class TestGateRegressions(unittest.TestCase):
    def test_empty_usage_is_blocked(self):
        self.assertEqual(verify([])['status'], 'BLOCKED')

    def test_missing_whole_document_is_blocked(self):
        self.assertEqual(verify(usage(), expected_doc_ids=['doc','missing'])['status'], 'BLOCKED')

    def test_strict_valid_usage(self):
        self.assertEqual(verify(usage(), expected_doc_ids=['doc'], require_provenance=True)['status'], 'PASS')

    def test_invalid_integer_identity_fails_without_crash(self):
        x=usage();x[0]['pass_index']='oops'
        self.assertEqual(verify(x)['status'], 'BLOCKED')

    def test_invalid_token_type_fails_without_crash(self):
        x=usage();x[0]['input_tokens']='1000'
        self.assertEqual(verify(x)['status'], 'BLOCKED')

    def test_boolean_tokens_rejected(self):
        x=usage();x[0]['input_tokens']=True
        self.assertEqual(verify(x)['status'], 'BLOCKED')

    def test_auxiliary_bound_enforced(self):
        x=usage();x[0]['auxiliary_tokens_upper_bound']=769
        self.assertEqual(verify(x, require_provenance=True)['status'], 'BLOCKED')

    def test_demo_bytes_must_match(self):
        x=usage();x[0]['demo_sha256']='f'*64
        self.assertEqual(verify(x, require_provenance=True)['status'], 'BLOCKED')

    def test_estimates_cannot_pass_as_actual_usage(self):
        x=usage();x[0]['token_count_source']='estimate'
        self.assertEqual(verify(x, require_provenance=True)['status'], 'BLOCKED')

    def test_failed_call_blocks_comparability(self):
        x=usage();x[0]['status']='error'
        self.assertEqual(verify(x, require_provenance=True)['status'], 'BLOCKED')

    def test_empty_provenance_is_blocked(self):
        x=usage();x[0]['model_version']=''
        self.assertEqual(verify(x)['status'], 'BLOCKED')

    def test_different_served_models_blocked(self):
        x=usage();x[0]['served_model']='another'
        self.assertEqual(verify(x, require_provenance=True)['status'], 'BLOCKED')

    def test_duplicate_selected_ids_are_rejected(self):
        with self.assertRaises(ValueError):
            prepare_view(document(), 'A', ['a','a'])

    def test_record_count_mismatch_not_hidden_by_zip(self):
        d=document();a,r=prepare_view(d,'A'),prepare_view(d,'R')
        r.auxiliary_records=[]
        with self.assertRaises(ValueError):validate_pair_view(d,a,r)

    def test_candidate_mutation_rejected(self):
        d=document();a,r=prepare_view(d,'A'),prepare_view(d,'R')
        r.original_candidates=[]
        with self.assertRaises(ValueError):validate_pair_view(d,a,r)

    def test_raw_gold_layers_rejected(self):
        for k in ('coref_clusters','coref_instance','causal_origin','causal_frame','plot_links'):
            raw=dict(doc_id='d',text='<a X>',tokens=['X'],mentions=['a'],spans={'a':[0]},candidates=[])
            raw[k]=[]
            with self.assertRaises(ValueError):Document.from_public_json(raw)

    def test_missing_entire_gold_document_fails(self):
        with self.assertRaises(ValueError):
            evaluate([dict(doc_id='d',positive_pairs=[])],
                     [dict(doc_id='d',arm='A',predicted_pairs=[])],
                     {'d':set(),'missing':set()}, {})

    def test_parser_accepts_only_json_list_or_fence(self):
        self.assertEqual(parse_response('```json\n[]\n```'), [])
        for content in ('explanation []','{}','invalid',None):
            with self.assertRaises((ValueError,TypeError)):parse_response(content)

    def test_utf8_bound_counts_non_ascii_bytes(self):
        self.assertGreater(byte_bound([{'content':'é'}]), byte_bound([{'content':'e'}]))


class TestFrozenRealRendering(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory()
        cls.docs,cls.jobs,cls.views,cls.checks=prepare(Path(cls.tmp.name))

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def test_DEV44_has_616_planned_slots(self):
        self.assertEqual(len(self.docs),44)
        self.assertEqual(len(self.jobs),616)
        self.assertEqual(sum(len(d.candidates) for d in self.docs),10916)

    def test_first_document_passes_conservative_preflight(self):
        self.assertTrue(self.checks[0]['input_preflight_within_cap'])
        self.assertGreater(self.checks[0]['selected_after_conservative_bound'],0)

    def test_every_registry_within_conservative_auxiliary_cap(self):
        self.assertTrue(all(j['auxiliary_tokens_upper_bound']<=768 for j in self.jobs))

    def test_each_paired_slot_has_identical_base_and_demo_bytes(self):
        slots={}
        for j in self.jobs:slots.setdefault((j['doc_id'],j['pass_index']),{})[j['arm']]=j
        for pair in slots.values():
            for field in ('base_prompt_sha256','demo_sha256','demo_id'):
                self.assertEqual(pair['A'][field],pair['R'][field])
            self.assertNotEqual(pair['A']['full_prompt_sha256'],pair['R']['full_prompt_sha256'])

    def test_demonstrations_outside_each_topic(self):
        self.assertTrue(all(j['demo_id'].split('_')[0] != j['doc_id'].split('_')[0] for j in self.jobs))

    def test_document_source_bytes_unchanged(self):
        source={d.doc_id:d.text for d in self.docs}
        self.assertTrue(all(v['document_text']==source[v['doc_id']] for v in self.views))
