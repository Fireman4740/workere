import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from steering.method_vote import combine, score_dataset
from steering import ap_campaign as ap, ap_protocol as p, campaign_blocks as c
from steering import validation_protocol as v, campaign_adjudicate as a
from steering.test_ap_campaign import fixture


class VoteTests(unittest.TestCase):
    def run_panel(self, base, checks):
        pair = ('a', 'b')
        original = {'CTB': {'d': {pair} if base else set()}}
        views = {('CTB', 'd', o): {pair: checks[o]} for o in (0, 1)}
        judgments = [dict(dataset='CTB', doc_id='d', pair=pair, reverse=bool(i),
                         status='valid' if b is not None else 'invalid_answer',
                         label='A' if b else 'N') for i, b in enumerate(checks[2:])]
        systems, _, _ = combine(original, {'CTB': {'d': [pair]}}, views, judgments)
        return {name: bool(pred['CTB']['d']) for name, pred in systems.items()}

    def test_majority_can_add_without_unanimity(self):
        r = self.run_panel(False, [True, True, True, False])
        self.assertTrue(r['five_view_majority'])
        self.assertFalse(r['four_view_consensus'])
        self.assertFalse(r['strict_add_majority_remove'])

    def test_asymmetric_filter_removes_three_rejections(self):
        r = self.run_panel(True, [False, False, False, True])
        self.assertFalse(r['five_view_majority'])
        self.assertTrue(r['four_view_consensus'])
        self.assertFalse(r['strict_add_majority_remove'])

    def test_missing_is_not_a_negative_vote(self):
        for baseline in (True, False):
            r = self.run_panel(baseline, [not baseline, not baseline, not baseline, None])
            for name in ('five_view_majority', 'four_view_consensus', 'strict_add_majority_remove'):
                self.assertEqual(r[name], baseline)

    def test_unanimity_and_even_tie(self):
        for baseline in (True, False):
            r = self.run_panel(baseline, [not baseline]*4)
            self.assertTrue(all(v == (not baseline) for v in r.values()))
            r = self.run_panel(baseline, [True, False, True, False])
            self.assertTrue(all(v == baseline for v in r.values()))

    def test_cached_pipeline_replays_scores_and_keeps_empty_documents(self):
        selected, docs, rules, base = fixture()
        selected = {'CTB': selected['CTB']}
        base = {'CTB': base['CTB']}
        jobs = v.screen_jobs(selected, v.public_docs(docs), rules)
        rows = [dict(key=j['key'], status='valid', answer='{"0":"A","1":"N"}') for j in jobs]
        judgments = [dict(dataset='CTB', doc_id=d, pair=['a','b'], reverse=reverse,
                          status='valid', label='A') for d in selected['CTB'] for reverse in (False,True)]
        repair = {'CTB': {d: {('a','b')} for d in selected['CTB']}}
        prior = dict(selected=selected, run_id='source', policies={name:p.metrics(pred,base,docs)['CTB']
                    for name,pred in [('A_vote',base),('C_repair',repair)]})
        with tempfile.TemporaryDirectory() as tmp:
            source, out = Path(tmp)/'source', Path(tmp)/'out'
            (source/'screen').mkdir(parents=True)
            (source/'repair').mkdir()
            ap.save(source/'report.json',prior)
            ap.save(source/'terminal.json',dict(status='complete', report_sha256=c.sha(source/'report.json')))
            ap.save(source/'manifest.json',{})
            ap.dump_predictions(source/'A_vote.json',base)
            ap.dump_predictions(source/'C_repair.json',repair)
            a.write_jsonl(source/'screen/requests.jsonl',jobs)
            a.write_jsonl(source/'screen/responses.jsonl',rows)
            a.write_jsonl(source/'repair/judgments.jsonl',judgments)
            with patch('steering.method_vote.c.inputs',return_value=(docs['CTB'],{},docs['ESC'],rules)), patch('builtins.print'):
                report=score_dataset('CTB',source,out)
            self.assertEqual(report['policies']['A_vote']['documents'],3)
            self.assertEqual(report['policies']['five_view_majority']['f1'],100)
            self.assertEqual(report['coverage']['candidate_pairs'],6)
            self.assertEqual(report['coverage']['five_view_complete'],3)
            self.assertEqual(report['new_model_calls'],0)


if __name__ == '__main__':
    unittest.main()
