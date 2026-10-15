"""Configuration expansion, execution receipts, comparison and offline probability replay."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from benchmark.campaign import execute_many
from benchmark.comparison import compare_runs, markdown_table
from benchmark.configuration import expand_campaign, expand_sweep, load_experiment, resolve_config
from benchmark.io import read_json, read_jsonl, write_json
from benchmark.suite import freeze
from pipelines.jev.analysis import analyze, rethreshold
from pipelines.jev.client import DecisionClient
from pipelines.jev.state_builder import build_state, choice_question, noul_question
from benchmark.protocols import public_inputs
from tests.test_benchmark import config, rows
from tests.test_framework_protocols import component


class Experiments(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.suite = freeze(config(), rows())
        write_json(self.root / 'suite.json', self.suite)
        self.cfg = {'schema': 'workere-experiment/v1', 'name': 'mock-decision', 'seed': 42,
            'dataset': {'name': 'synthetic', 'split': 'train', 'suite': 'suite.json'},
            'protocol': 'relation_gold_mentions', 'models': [{'id': 'typesafe/jev-1.14', 'version': '1.14'}],
            'plugins': ['pipelines.jev.pipeline'], 'pipeline': {'entity_extractor': component('reference'),
                'candidate_generator': component('historical'), 'relation_predictor': component('jev_decision',
                    threshold=.5, variant='noul', context_style='document',
                    client={'model': 'typesafe/jev-1.14', 'model_version': '1.14', 'mode': 'mock',
                            'cache_path': 'cache.sqlite', 'max_calls': 100})},
            'evaluation': {'primary_metric': 'binary', 'bootstrap_samples': 30},
            'execution': {'allow_network': False, 'allow_paid': False, 'error_policy': 'raise'}, 'output_dir': 'runs'}
        self.source = self.root / 'experiment.yaml'
        self.save()

    def save(self):
        self.source.write_text(yaml.safe_dump(self.cfg))

    def execute(self):
        return execute_many([load_experiment(self.source)])['runs'][0]

    def test_seed_campaign_sweep_and_dry_run_have_no_model_effects(self):
        self.cfg.pop('seed'); self.cfg['seeds'] = [7, 42]; self.save()
        campaign = self.root / 'campaign.yaml'
        campaign.write_text(yaml.safe_dump({'schema': 'workere-campaign/v1', 'experiments': [
            {'config': 'experiment.yaml', 'overrides': {'name': 'another'}}]}))
        configs = list(expand_campaign(campaign))
        self.assertEqual([c.data['seed'] for c in configs], [7, 42])
        self.assertEqual(configs[0].data['name'], 'another')
        self.assertEqual(configs[0].data['dataset']['suite'], str((self.root / 'suite.json').resolve()))
        sweep = self.root / 'sweep.yaml'
        sweep.write_text(yaml.safe_dump({'schema': 'workere-sweep/v1', 'base': 'experiment.yaml',
            'grid': {'pipeline.relation_predictor.params.threshold': [.2, .8]}}))
        with patch('benchmark.engine.build_pipeline', side_effect=AssertionError('dry run cannot instantiate models')):
            plan = execute_many(expand_sweep(sweep), dry_run=True)
        self.assertEqual(plan['count'], 4)
        self.assertFalse((self.root / 'runs').exists())
        invalid = copy.deepcopy(self.cfg); invalid['seed'] = 99
        with self.assertRaises(ValueError):
            resolve_config(invalid, self.source)

    def test_mock_noul_then_choice_and_context_use_same_scorer(self):
        first = self.execute()
        report = read_json(Path(first['run_dir']) / 'metrics.json')
        with patch('pipelines.jev.client.DecisionClient._send', side_effect=AssertionError('offline operation')):
            analysis = analyze(first['run_dir'], self.root / 'analysis')
            replay = rethreshold(first['run_dir'], .5, self.root / 'threshold')
        self.assertEqual(replay['summary'], report['summary'])
        self.assertEqual(analysis['candidates'], 12)
        self.assertEqual(sum(analysis['confusion_matrix'].values()), 12)
        self.assertEqual(analysis['api_calls'], 0)
        self.assertEqual(analysis['baselines'][0]['summary']['binary']['f1'], 0)
        self.assertEqual(len(analysis['threshold_sensitivity']), 21)
        journal = read_jsonl(Path(first['run_dir']) / 'decisions.jsonl')
        self.assertIn('positive_probability', journal[0])
        self.assertIn('class_confidence', journal[0])
        for variant, style in [('choice', 'document'), ('noul', 'explicit_context')]:
            self.cfg['name'] = f'mock-{variant}-{style}'
            self.cfg['pipeline']['relation_predictor']['params'].update(variant=variant, context_style=style)
            self.save(); result = self.execute()
            prediction = read_jsonl(Path(result['run_dir']) / 'predictions.jsonl')[0]
            self.assertEqual(prediction['metadata']['decision_type'], variant)
            self.assertEqual(prediction['metadata']['context_style'], style)
            self.assertEqual(len(prediction['metadata']['decision_scores']), len(prediction['relations']))

    def test_comparison_rejects_different_suites_and_aggregates_seeds(self):
        first = self.execute()
        self.cfg['seed'] = 7; self.save(); second = self.execute()
        comparison = compare_runs([first['run_dir'], second['run_dir']], samples=20)
        self.assertEqual(len(comparison['groups']), 1)
        self.assertEqual(len(comparison['groups'][0]['seeds']), 2)
        self.assertIn('| Run |', markdown_table(comparison))
        path = Path(second['run_dir']) / 'metrics.json'
        metrics = read_json(path); metrics['suite_fingerprint'] = 'different'; path.unlink(); write_json(path, metrics)
        with self.assertRaisesRegex(ValueError, 'incomparable'):
            compare_runs([first['run_dir'], second['run_dir']])

    def test_choice_rejects_span_before_transport_and_preserves_probabilities(self):
        with DecisionClient(model='inception/mercury-decide:free', model_version='test',
                            cache_path=self.root / 'choice.sqlite') as client:
            answer = client.decide('Rain caused flooding', choice_question())['response']['answers']['relation']
            self.assertAlmostEqual(sum(answer['probabilities'].values()), 1)
        client = DecisionClient(model='respan/span-01-lite', model_version='test', cache_path=self.root / 'span.sqlite')
        self.addCleanup(client.close)
        with self.assertRaisesRegex(ValueError, 'noul only'):
            client.decide('Rain caused flooding', choice_question())

    def test_failure_retains_each_successful_decision_and_error_in_journal(self):
        original = DecisionClient._send
        n = 0
        def send(client, body):
            nonlocal n
            n += 1
            if n >= 2:
                raise ValueError('synthetic provider error')
            return original(client, body)
        with patch.object(DecisionClient, '_send', send):
            failed = self.execute()
        self.assertEqual(failed['status'], 'failed')
        journal = read_jsonl(Path(failed['run_dir']) / 'decisions.jsonl')
        self.assertEqual(len(journal), 2)
        self.assertIsNotNone(journal[0]['positive_probability'])
        self.assertEqual(journal[1]['error']['type'], 'ValueError')
        with self.assertRaisesRegex(ValueError, 'completed'):
            analyze(failed['run_dir'], self.root / 'invalid-analysis')


if __name__ == '__main__':
    unittest.main()
