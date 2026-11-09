"""Scientific decision, cache and transport invariants without model calls."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from steering import validation_protocol as v
from steering import validation_campaign as run
from steering import ap_campaign as ap
from steering import ap_protocol as p
from steering.test_ap_campaign import fixture
from steering.test_campaign_api import Tokenizer


class ValidationTests(unittest.TestCase):
    def test_decimal_event_surface_uses_verbatim_context_and_missing_tag_still_fails(self):
        text = 'A <23 - 6 . 0> <10 quake> happened. A <24 - 6 . 1> <9 temblor> followed.'
        self.assertEqual(v.source_window(text, {'23', '24'}), text)
        with self.assertRaises(ValueError): v.source_window(text, {'23', 'absent'})
        docs = {'ESC': {'x': dict(id='x', text=text, pair_list=[['23', '24'], ['24', '23']])}}
        routes = [dict(dataset='ESC', doc_id='x', pair=['23', '24'], baseline=False)]
        rows, _ = run.repair_jobs(routes, docs, {'ESC': 'Rules'})
        self.assertTrue(all(row['context'] == text for row in rows))

    def test_repaired_budget_preserves_old_attempts_and_excludes_idle(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            ap.save(out/'budget.json', dict(start=0, max_hours=12))
            ap.save(out/'wall-old.json', dict(start=0, end=900, seconds=900))
            with patch.dict('os.environ', {'ORX_RUN_ID': 'new'}), patch.object(run.time, 'time', return_value=100000):
                self.assertEqual(run.attempt_deadline(out, 'MAVEN'), 100000+12*3600-900)
                self.assertEqual(run.attempt_deadline(out, 'MAVEN'), 100000+12*3600-900)
            self.assertEqual(ap.read(out/'wall-old.json')['seconds'], 900)

    def test_filter_requires_two_valid_negatives(self):
        base = {'CTB': {'d': {('a', 'b')}}}
        for labels, removed in [((False, False), True), ((False, None), False),
                                ((False, True), False), ((None, None), False)]:
            views = {('CTB', 'd', i): {('a', 'b'): x} for i, x in enumerate(labels) if x is not None}
            systems = v.combine(base, views, {})
            self.assertEqual(('a', 'b') not in systems['B_filter']['CTB']['d'], removed)
            self.assertEqual(systems['A_vote'], base)

    def test_combination_uses_original_baseline_without_readding_filtered_link(self):
        base = {'CTB': {'d': {('a', 'b')}}}
        views = {('CTB', 'd', i): {('a', 'b'): False} for i in (0, 1)}
        decisions = {('CTB', 'd', ('a', 'b')): 'A', ('CTB', 'd', ('a', 'c')): 'B'}
        systems = v.combine(base, views, decisions)
        self.assertEqual(systems['D_filter_add']['CTB']['d'], {('a', 'c')})
        self.assertEqual(systems['C_repair']['CTB']['d'], {('a', 'b'), ('a', 'c')})
        self.assertEqual(base['CTB']['d'], {('a', 'b')})

    def test_requests_and_routing_are_gold_blind(self):
        selected, docs, rules, base = fixture()
        changed = copy.deepcopy(docs)
        for values in changed.values():
            for row in values.values():
                row['gold'] = ['POISON']; row['coref'] = [['a', 'c']]
        jobs = v.screen_jobs(selected, v.public_docs(docs), rules)
        self.assertEqual(jobs, v.screen_jobs(selected, v.public_docs(changed), rules))
        self.assertNotIn('POISON', json.dumps(jobs))
        views = {('CTB', 'c0', 0): {('a', 'b'): True}}
        self.assertEqual(v.routes_for(selected, docs, base, views), v.routes_for(selected, changed, base, views))

    def test_screen_partition_and_original_c2_order(self):
        import random
        selected, docs, rules, _ = fixture()
        jobs = v.screen_jobs(selected, docs, rules)
        for ds, ids in selected.items():
            for doc_id in ids:
                for order in (0, 1):
                    panel = [j for j in jobs if j['meta']['dataset'] == ds and j['meta']['doc_id'] == doc_id and j['meta']['order'] == order]
                    pairs = [tuple(x) for j in panel for x in j['meta']['pairs']]
                    self.assertEqual(sorted(pairs), run.c.pairs(docs[ds][doc_id]))
                    self.assertEqual(len(pairs), len(set(pairs)))
                    self.assertTrue(all((j['thinking'], j['answer_tokens']) == (4096, 384) for j in panel))
                    block = panel[0]['meta']['pairs']; indices = list(range(len(block)))
                    if order: random.Random(f'{run.c.SEED}::{ds}::{doc_id}::0').shuffle(indices)
                    expected = run.c.make_messages(rules[ds], run.c.windows(docs[ds][doc_id]['text'], {e for x in block for e in x}), block, indices)
                    self.assertEqual(panel[0]['messages'], expected)

    def test_invalid_final_cannot_be_read_from_thought(self):
        selected, docs, rules, _ = fixture()
        jobs = v.screen_jobs({'CTB': ['c0']}, docs, rules)
        outputs = {j['key']: dict(status='invalid_answer', answer='broken', thought='{"0":"A","1":"A"}') for j in jobs}
        self.assertEqual(v.views_from(jobs, outputs), {})

    def test_repair_consensus_requires_canonical_direction(self):
        selected, docs, rules, base = fixture()
        routes = [dict(dataset='CTB', doc_id='c0', pair=['a', 'b'], baseline=False)]
        rows, jobs = run.repair_jobs(routes, docs, rules)
        outputs = {j['key']: dict(status='valid', answer=json.dumps({'label': 'B' if row['reverse'] else 'A'}))
                   for row, j in zip(rows, jobs)}
        decisions, _ = run.repair_decisions(rows, jobs, outputs)
        self.assertEqual(decisions[('CTB', 'c0', ('a', 'b'))], 'A')
        reversed_row = next(j for row, j in zip(rows, jobs) if row['reverse'])
        outputs[reversed_row['key']]['answer'] = '{"label":"A"}'
        decisions, _ = run.repair_decisions(rows, jobs, outputs)
        self.assertIsNone(decisions[('CTB', 'c0', ('a', 'b'))])

    def test_import_only_exact_prompts_and_count_only_new_cost(self):
        selected, docs, rules, _ = fixture()
        jobs = v.screen_jobs({'CTB': ['c0']}, docs, rules)
        row = dict(status='valid', answer='{"0":"N","1":"N"}', thinking_forced=False,
                   prompt_tokens=2, total_input_tokens=4, think_tokens=2, answer_tokens=1)
        catalog = {v.cache_identity(jobs[0]): (row, '/old/native-c2')}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'screen'
            run.import_responses(jobs, out, catalog)
            run.import_responses(jobs, out, catalog)
            self.assertEqual(len(run.c.jsonl(out/'responses.jsonl')), 1)
            self.assertEqual(run.incremental_cost(Path(tmp)), {'reused_responses': 1})
            modified = dict(jobs[0], answer_tokens=512)
            self.assertNotEqual(v.fingerprint(modified), v.fingerprint(jobs[0]))

    def test_full_denominator_and_split_report(self):
        selected, docs, _, base = fixture()
        ds = 'CTB'; selected = {ds: selected[ds]}; base = {ds: base[ds]}
        systems = v.combine(base, {}, {})
        metrics, splits = run.evaluate(systems, base, docs, selected, ['c0'])
        self.assertEqual(metrics['A_vote']['documents'], 3)
        self.assertEqual(metrics['A_vote']['fn'], 6)
        self.assertEqual(splits['extension']['B_filter']['documents'], 2)

    def test_document_parser_rejects_partial_and_duplicate_json(self):
        self.assertEqual(v.document_parse('{"relations":[]}'), {'relations': []})
        for text in ['{"relations":[],"relations":[]}', '{"relations":[]',
                     '{"relations":[{"pair":"a","label":"causes"}]}',
                     '{"relations":[{"pair":"a,b","label":"unknown"}]}']:
            self.assertIsNone(v.document_parse(text))

    def test_document_vote_has_fixed_denominator_and_candidate_filter(self):
        docs = {'MAVEN': {'d': dict(pair_list=[['a', 'b'], ['b', 'a']])}}
        jobs = [dict(key=str(i), meta=dict(doc_id='d', pass_id=i)) for i in range(5)]
        outputs = {str(i): dict(status='valid' if i < 2 else 'invalid_answer',
                    answer='{"relations":[{"pair":"a,b","label":"causes"},{"pair":"x,y","label":"causes"}]}') for i in range(5)}
        baseline, quality = v.document_vote({'MAVEN': ['d']}, docs, jobs, outputs)
        self.assertEqual(baseline['MAVEN']['d'], set())
        self.assertEqual(quality['invalid_reads'], 3)
        self.assertEqual(quality['out_of_universe_items'], 2)

    def test_document_transport_final_only_and_choice_reordering(self):
        engine = run.DocumentEngine.__new__(run.DocumentEngine)
        engine.tokenizer = Tokenizer(); engine.model = 'qwen'
        seen = []
        def call(body):
            seen.append(body)
            return ({'choices': [dict(index=i, text='{"relations":[]}' if len(seen) == 2 else 'reason',
                      finish_reason='stop', stop_reason='</think>', token_ids=[1]) for i in reversed(range(2))]}, False, .1)
        engine.call = call
        jobs = [ap.job([dict(role='user', content=str(i))], v.document_schema(), 64, 128, dict(i=i)) for i in range(2)]
        rows, cost = engine.batch(jobs)
        self.assertEqual([r['key'] for r in rows], [j['key'] for j in jobs])
        self.assertTrue(all(r['status'] == 'valid' for r in rows))
        self.assertIn('\n</think>\n\n', bytes(seen[1]['prompt'][0]).decode())
        self.assertEqual(seen[1]['structured_outputs']['json'], v.document_schema())

    def test_maven_preparation_uses_disjoint_train_demos_and_is_eval_gold_blind(self):
        import pyarrow.parquet as parquet
        def row(i, train=False):
            return dict(id=f'{"tr" if train else "te"}{i}', tokens=['rain', 'flood'], spans=[[0], [1]],
                mentions=['a', 'b'], sentences=[[0, 2]], pair_list=[[0, 1], [1, 0]],
                text=f'{"Training" if train else "Test"} {i}: <a rain> caused <b flood>.',
                annots='<a rain> causes <b flood> <b flood> causedby <a rain>')
        test_rows = [row(i) for i in range(710)]
        train_rows = [row(i, True) for i in range(8)]
        class Table:
            def __init__(self, values): self.values = values
            def to_pylist(self): return self.values
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); old = root/'old'; (old/'AP0').mkdir(parents=True)
            ap.save(old/'AP0/api_contract.json', {'same': 'contract'})
            train = root/'train'; test = root/'test'; train.write_text('train'); test.write_text('test')
            def download(repo, filename, **kwargs):
                self.assertEqual(kwargs['revision'], run.MAVEN_REVISION)
                return str(train if 'train-' in filename else test)
            def read_table(path): return Table(train_rows if path == train else test_rows)
            with patch('huggingface_hub.hf_hub_download', side_effect=download), patch.object(parquet, 'read_table', side_effect=read_table):
                out = root/'one'; out.mkdir()
                selected, docs, rules, jobs = run.prepare_maven(root, old, out)
                for r in test_rows: r['annots'] = ''
                second = root/'two'; second.mkdir()
                new_selected, _, _, new_jobs = run.prepare_maven(root, old, second)
                self.assertEqual(selected, new_selected)
                self.assertEqual(jobs, new_jobs)
                self.assertEqual(len(jobs), 60)
                self.assertTrue(all(j['meta']['demo_id'].startswith('tr') for j in jobs))
                self.assertEqual(len(docs['MAVEN']), 12)


if __name__ == '__main__': unittest.main()
