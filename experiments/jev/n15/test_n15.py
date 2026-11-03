import copy
import json
import unittest
from pipelines.jev import n15


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.spec, self.rows, self.docs = n15.preflight()
        self.events = []
        for i, r in enumerate(self.rows):
            d = self.docs[r['doc']]
            a, b = sorted(d['pair_list'][0])
            # Both orientations in a single response must still count as one vote.
            content = json.dumps([{'pair': a+','+b, 'label': 'causes'},
                                  {'pair': b+','+a, 'label': 'causedby'}])
            response = dict(id=str(i), model=self.spec['model'], choices=[dict(
                finish_reason='stop', message=dict(content=content))])
            self.events.append(dict(logical_id=r['logical_id'], request=r['body'],
                                    response=response, error=None))

    def test_full_pair_coverage_and_one_vote_per_read(self):
        decisions = n15.freeze(self.rows, self.events, self.docs)
        self.assertEqual(len(decisions), 378)
        self.assertTrue(all(d['votes'] in (0, 3) for d in decisions))

    def test_missing_response_is_not_negative(self):
        with self.assertRaises(AssertionError):
            n15.freeze(self.rows, self.events[:-1], self.docs)

    def test_truncation_is_not_negative(self):
        self.events[0]['response']['choices'][0]['finish_reason'] = 'length'
        with self.assertRaises(AssertionError):
            n15.freeze(self.rows, self.events, self.docs)

    def test_wrong_prompt_cannot_be_reused(self):
        self.events[0]['request'] = copy.deepcopy(self.events[0]['request'])
        self.events[0]['request']['max_tokens'] = 65536
        with self.assertRaises(AssertionError):
            n15.freeze(self.rows, self.events, self.docs)

    def test_duplicate_response_rejected(self):
        self.events[0]['response']['id'] = self.events[1]['response']['id']
        with self.assertRaises(AssertionError):
            n15.freeze(self.rows, self.events, self.docs)

    def test_micro_score_from_counts(self):
        gold = [dict(doc='a', a='1', b='2', gold=True),
                dict(doc='b', a='1', b='2', gold=False),
                dict(doc='b', a='1', b='3', gold=True)]
        decisions = [dict(condition=c, doc=r['doc'], a=r['a'], b=r['b'], positive=r['b']=='2')
                     for c in ('medium', 'no_thinking') for r in gold]
        for s in n15.score(decisions, gold).values():
            self.assertEqual((s['TP'], s['FP'], s['FN'], s['F1']), (1, 1, 1, 50))


if __name__ == '__main__':
    unittest.main()
