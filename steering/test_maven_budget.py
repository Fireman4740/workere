"""Regression checks for pairing, frozen traces, invalid reads and resume."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from steering import maven_budget as mb
from steering.test_maven_pool import DOC, _row


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return 'prompt<think>'

    def encode(self, text, **kwargs):
        return list(text.encode())


class Client:
    def __init__(self, choices):
        self.choices, self.bodies = list(choices), []

    def call(self, path, body, **kwargs):
        self.bodies.append(body)
        return {'choices': [dict(index=0, **self.choices.pop(0))]}, False, .1


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = mb.config('steering/maven_budget_config.json')
        self.cfg.update(arm='A', thinking_tokens=20480, answer_tokens=2048,
                        source_arm=None, campaign_root=self.tmp.name)
        self.manifest = dict(engine_sha256='engine', jobs_sha256='jobs')
        self.job = dict(key='j', doc_id='d', pass_id=0, demo_id='train', messages=[])

    def reader(self, cfg, client, source=None):
        reader = object.__new__(mb.Reader)
        reader.cfg, reader.manifest = cfg, self.manifest
        reader.out = mb.output_dir(cfg)
        reader.source = source or reader.out
        reader.tok, reader.client = Tokenizer(), client
        return reader

    @patch.dict(os.environ, ORX_RUN_ID='test-run', ORX_COMMIT='test-commit')
    def test_exact_trace_reuse_answer_budget_and_resume(self):
        first = Client([dict(text='reason', finish_reason='stop', stop_reason='</think>'),
                        dict(text='{"relations": []}', finish_reason='stop')])
        a = self.reader(self.cfg, first)
        row, hit = a.one(self.job)
        self.assertFalse(hit)
        self.assertEqual(row['status'], 'valid')
        self.assertFalse(row['thinking_forced'])
        self.assertEqual(a.one(self.job), (row, True))
        self.assertEqual(len(first.bodies), 2)
        bcfg = dict(self.cfg, arm='B', source_arm='A', answer_tokens=4096)
        second = Client([dict(text='{"relations": []}', finish_reason='stop')])
        b = self.reader(bcfg, second, a.out)
        brow, _ = b.one(self.job)
        self.assertEqual(len(second.bodies), 1)
        self.assertEqual(second.bodies[0]['prompt'], first.bodies[1]['prompt'])
        self.assertEqual(second.bodies[0]['max_tokens'], 4096)
        self.assertEqual(brow['trace_sha256'], row['trace_sha256'])
        self.assertTrue(brow['trace_reused'])
        self.assertNotEqual(brow['contract'], row['contract'])

    @patch.dict(os.environ, ORX_RUN_ID='test-run', ORX_COMMIT='test-commit')
    def test_truncated_answer_is_invalid_even_if_json_parses(self):
        client = Client([dict(text='reason', finish_reason='length'),
                         dict(text='{"relations": []}', finish_reason='length')])
        row, _ = self.reader(self.cfg, client).one(self.job)
        self.assertEqual(row['status'], 'invalid_answer')
        self.assertTrue(row['thinking_forced'])

    def test_consumer_never_regenerates_missing_trace(self):
        cfg = dict(self.cfg, arm='B', source_arm='A', answer_tokens=4096)
        client = Client([])
        with self.assertRaisesRegex(ValueError, 'regeneration is forbidden'):
            self.reader(cfg, client, mb.output_dir(self.cfg)).one(self.job)
        self.assertFalse(client.bodies)

    def test_trace_contract_excludes_answer_but_pins_engine_and_thinking(self):
        contract = mb.trace_contract(self.cfg, self.manifest, self.job)
        self.assertEqual(contract, mb.trace_contract(dict(self.cfg, answer_tokens=4096), self.manifest, self.job))
        self.assertNotEqual(contract, mb.trace_contract(dict(self.cfg, thinking_tokens=40960), self.manifest, self.job))
        self.assertNotEqual(contract, mb.trace_contract(self.cfg, dict(self.manifest, engine_sha256='other'), self.job))
        data = {'final_prompt_ids': [1, 2, 3]}
        record = dict(contract=contract, data=data, sha256=mb.api.digest(data))
        mb.validate_trace(record, contract)
        data['final_prompt_ids'].append(4)
        with self.assertRaises(ValueError):
            mb.validate_trace(record, contract)

    def test_fixed_vote_and_distance_metrics(self):
        jobs = [dict(self.job, key=str(i), pass_id=i) for i in range(5)]
        rows = {str(i): _row(str(i), [('e1', 'e2')]) for i in range(3)}
        rows.update({str(i): _row(str(i), [], 'invalid_answer') for i in (3, 4)})
        structure = {'d': {'sentence_of': {'e1': 0, 'e2': 4, 'e3': 1}}}
        pred, counts, quality = mb.pairs_and_counts(jobs, rows, {'d': DOC}, structure, 3)
        self.assertEqual(pred['d'], {('e1', 'e2')})
        self.assertEqual(counts['all']['d'], dict(tp=1, fp=0, fn=0))
        self.assertEqual(counts['distance_ge4']['d'], counts['all']['d'])
        self.assertEqual(quality['invalid_reads'], 2)
        self.assertFalse(mb.pairs_and_counts(jobs, rows, {'d': DOC}, structure, 4)[0]['d'])

    def test_missing_read_prevents_completion(self):
        out = mb.output_dir(self.cfg)
        jobs = [self.job]
        manifest = dict(self.manifest, configuration=self.cfg, jobs_sha256=mb.api.digest(jobs))
        for name, value in [('manifest', manifest), ('jobs', jobs), ('documents', {'d': DOC}),
                            ('doc_structure', {})]:
            mb.atomic(out / (name+'.json'), value)
        with self.assertRaises(FileNotFoundError):
            mb.score(self.cfg)
        self.assertFalse((out / 'complete.json').exists())

    def test_source_requires_committed_completion_pin(self):
        cfg = dict(self.cfg, source_arm='A', source_completion_sha256=None)
        with self.assertRaisesRegex(ValueError, 'must be pinned'):
            mb.source_bundle(cfg, self.manifest, [self.job])

    def test_import_freezes_demo_associations_and_strips_gold(self):
        cfg = dict(self.cfg, documents=1, passes=5, historical_root=self.tmp.name)
        jobs = [dict(self.job, key=str(i), pass_id=i, demo_id=str(i), budget='long', gold=['secret'])
                for i in range(5)]
        Path(self.tmp.name, 'jobs.jsonl').write_text('\n'.join(json.dumps(j) for j in jobs))
        old = dict(selected=['d'], demo_ids=list(map(str, range(5))),
                   jobs_sha256=mb.api.digest([j['key'] for j in jobs]))
        got = mb.import_jobs(cfg, old, {'d': DOC})
        self.assertEqual(len(got), 5)
        self.assertTrue(all('gold' not in j for j in got))
        old['demo_ids'][0] = 'wrong'
        with self.assertRaises(ValueError):
            mb.import_jobs(cfg, old, {'d': DOC})

    def test_joint_bootstrap_preserves_pairing_and_interaction(self):
        reports = {a: {'per_document': {'vote_ge3': {'all': {
            'd': dict(tp=n, fp=0, fn=4-n)}}}} for a, n in zip('ABCD', [1, 2, 2, 4])}
        result = mb.paired_contrasts(reports, draws=100)
        f = {a: mb.micro_f1(list(r['per_document']['vote_ge3']['all'].values())) for a, r in reports.items()}
        self.assertAlmostEqual(result['D-B']['delta_f1'], f['D']-f['B'])
        self.assertAlmostEqual(result['(D-C)-(B-A)']['delta_f1'], f['D']-f['C']-f['B']+f['A'])
        self.assertEqual(result['D-B']['ci95'], [result['D-B']['delta_f1']]*2)
        reports['D']['per_document']['vote_ge3']['all']['other'] = dict(tp=0, fp=0, fn=1)
        with self.assertRaises(ValueError):
            mb.paired_contrasts(reports, draws=100)

    def test_immutable_file_refuses_replacement(self):
        p = Path(self.tmp.name, 'trace.json')
        mb.atomic(p, {'a': 1}, immutable=True)
        with self.assertRaises(ValueError):
            mb.atomic(p, {'a': 2}, immutable=True)
        self.assertEqual(mb.read(p), {'a': 1})


if __name__ == '__main__':
    unittest.main()
