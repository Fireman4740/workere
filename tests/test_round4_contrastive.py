import json
import unittest
import tempfile
from unittest.mock import patch
from pathlib import Path

from benchmark.data import parse_annotations
from benchmark.schema import Document, Entity
from pipelines.jev.contrastive import negative_pairs, select_demonstrations
from pipelines.jev.llm import LunaDocumentPredictor
from benchmark.interfaces import Context


class ContrastiveSafety(unittest.TestCase):
    def test_actual_messages_contain_both_labels_and_only_bank_annotations(self):
        bank = json.loads(Path('experiments/jev/round4/demo-bank.json').read_text())['documents']
        d = json.loads(Path('experiments/jev/round4/dev44-suite.json').read_text())['documents'][0]
        query = Document(d['id'], d['tokens'], d['sentences'])
        entities = [Entity(m, sp) for m, sp in zip(d['mentions'], d['spans'])]
        with tempfile.TemporaryDirectory() as tmp:
            predictor = LunaDocumentPredictor({'model': 'openai/gpt-6-luna:batch', 'model_version': 'gpt-6-luna-20260922',
                'endpoint': 'https://openrouter.ai/api/v1/batches', 'mode': 'mock', 'cache_path': str(Path(tmp)/'cache.sqlite'),
                'max_calls': 10, 'reasoning_effort': 'high', 'max_context_tokens': 1050000},
                'prompts/esl_standard_eci_lean_coref.yaml', 'unused', 'unused', 'unused', demo_selection='contrastive_structural')
            predictor.bank = bank
            captured = []
            def fake(messages, callbacks):
                captured.extend(messages)
                return [{'response': {'model': 'mock/gpt-6-luna-20260922', 'choices': [{'message': {'content': '[]'}}]}} for _ in messages]
            context = Context(42, 'relations_gold', query.id)
            with patch.object(predictor.client, 'decide_many', side_effect=fake):
                predictor.predict(query, entities, [tuple(pair) for pair in d['pair_list']], context)
            self.assertEqual(len(captured), 7)
            chosen = select_demonstrations(query, entities, bank)
            for messages, demo in zip(captured, chosen):
                answer = json.loads(messages[2]['content'])
                positive = {(r['pair'], r['label']) for r in answer if r['label'] != 'norel'}
                expected = {(a+','+b, lab) for a,lab,b in parse_annotations(demo['annots']) if lab != 'norel'}
                self.assertEqual(positive, expected)
                negative = {r['pair'] for r in answer if r['label'] == 'norel'}
                self.assertTrue(negative)
                self.assertTrue(all(b+','+a in negative for a,b in (p.split(',') for p in negative)))
            self.assertEqual(context.prediction_metadata['demonstration_selection'], 'contrastive_structural')
            predictor.close()

    def test_all_queries_have_seven_contrasting_demos_without_topic_leakage(self):
        bank = json.loads(Path('experiments/jev/round4/demo-bank.json').read_text())['documents']
        suite = json.loads(Path('experiments/jev/round4/dev44-suite.json').read_text())
        for d in suite['documents']:
            query = Document(d['id'], d['tokens'], d['sentences'])
            entities = [Entity(m, sp) for m, sp in zip(d['mentions'], d['spans'])]
            chosen = select_demonstrations(query, entities, bank)
            self.assertEqual(len(chosen), 7)
            self.assertEqual(len({r['id'] for r in chosen}), 7)
            self.assertEqual([r['id'] for r in chosen], [r['id'] for r in select_demonstrations(query, entities, bank)])
            for demo in chosen:
                self.assertNotEqual(demo['id'].split('_')[0], query.id.split('_')[0])
                self.assertNotIn(demo['id'], {r['id'] for r in suite['documents']})
                pos = {frozenset((a, b)) for a, lab, b in parse_annotations(demo['annots']) if lab != 'norel'}
                neg = negative_pairs(demo)
                self.assertTrue(pos)
                self.assertTrue(neg)
                self.assertLessEqual(len(neg), 4)
                self.assertTrue(all(frozenset(p) not in pos for p in neg))


if __name__ == '__main__':
    unittest.main()
