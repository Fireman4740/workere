import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from steering import maven_pool as mp


def _jobs(doc_ids):
    jobs = []
    for d in doc_ids:
        for budget, think in mp.BUDGETS.items():
            for i in range(5):
                jobs.append(dict(key=f'{d}-{budget}-{i}', doc_id=d, budget=budget, pass_id=i, thinking=think))
    return jobs


def _row(key, pairs, status='valid'):
    rel = [dict(pair=f'{x},{y}', label='causes') for x, y in pairs]
    return dict(key=key, status=status, answer=json.dumps({'relations': rel}))


DOC = dict(pair_list=[['e1', 'e2'], ['e2', 'e1'], ['e1', 'e3'], ['e3', 'e1'], ['e2', 'e3'], ['e3', 'e2']],
           gold=['e1,e2', 'e2,e1'])


class MavenPoolTests(unittest.TestCase):
    def test_vote_thresholds(self):
        reads = [{('a', 'b')}, {('a', 'b'), ('c', 'd')}, {('a', 'b')}, set(), {('c', 'd')}]
        self.assertEqual(mp.vote(reads, 3), {('a', 'b')})
        self.assertEqual(mp.vote(reads, 2), {('a', 'b'), ('c', 'd')})
        self.assertEqual(mp.vote([], 3), set())

    def test_pool_vote_uses_fixed_denominators_and_counts_invalid_as_empty(self):
        jobs = _jobs(['d'])
        rows = {}
        for j in jobs:  # e1-e2 in 3 short + 3 long reads; e1-e3 in 4 long reads only
            pairs = []
            if j['pass_id'] < 3:
                pairs.append(('e1', 'e2'))
            if j['budget'] == 'long' and j['pass_id'] < 4:
                pairs.append(('e1', 'e3'))
            rows[j['key']] = _row(j['key'], pairs)
        rows['d-short-4'] = dict(key='d-short-4', status='invalid_answer', answer='')
        del rows['d-long-4']
        systems, quality = mp.systems_from(jobs, rows, {'d': DOC})
        self.assertEqual(systems['short_vote5']['d'], {('e1', 'e2')})
        self.assertEqual(systems['long_vote5']['d'], {('e1', 'e2'), ('e1', 'e3')})
        self.assertEqual(systems['pool_vote10_ge6']['d'], {('e1', 'e2')})
        self.assertEqual(systems['pool_vote10_ge5']['d'], {('e1', 'e2')})
        self.assertEqual(quality['invalid_reads'], 2)
        self.assertEqual(quality['missing_reads'], 1)

    def test_out_of_universe_pairs_are_dropped(self):
        quality = mp.Counter()
        got = mp.reading_pairs(_row('k', [('e1', 'e2'), ('e1', 'zz')]), {('e1', 'e2')}, quality)
        self.assertEqual(got, {('e1', 'e2')})
        self.assertEqual(quality['out_of_universe_items'], 1)

    def test_counts_are_symmetric_presence(self):
        per = mp.per_doc_counts({'d': {('e1', 'e2'), ('e1', 'e3')}}, {'d': DOC})
        self.assertEqual(per['d'], dict(tp=2, fp=2, fn=0))

    def test_shards_partition_documents_and_put_long_reads_first(self):
        jobs = _jobs([f'd{i}' for i in range(7)])
        shards = [mp.shard_jobs(jobs, s, 3) for s in range(3)]
        keys = [j['key'] for s in shards for j in s]
        self.assertEqual(sorted(keys), sorted(j['key'] for j in jobs))
        docs = [{j['doc_id'] for j in s} for s in shards]
        self.assertFalse(docs[0] & docs[1] or docs[1] & docs[2])
        self.assertTrue(all(s[0]['budget'] == 'long' for s in shards))

    def test_selection_is_stratified_seeded_and_skips_too_long(self):
        test = [SimpleNamespace(id=f'd{i:03d}') for i in range(400)]
        lengths = {d.id: (None if i % 10 == 0 else i) for i, d in enumerate(test)}
        a = mp.select_docs(test, lengths)
        b = mp.select_docs(test, lengths)
        self.assertEqual([d.id for d in a], [d.id for d in b])
        self.assertEqual(len({d.id for d in a}), mp.N_DOCS)
        self.assertTrue(all(lengths[d.id] is not None for d in a))

    def test_read_jsonl_ignores_partial_last_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'r.jsonl'
            p.write_text('{"key": "a"}\n{"key": "b"}\n{"key": ')
            self.assertEqual([r['key'] for r in mp.read_jsonl(p)], ['a', 'b'])


if __name__ == '__main__':
    unittest.main()
