import importlib.util
import json
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('n4', Path(__file__).with_name('run.py'))
n4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(n4)


class Checks(unittest.TestCase):
    def test_pairing_and_no_labels(self):
        root = Path(__file__).with_name('frozen')
        data = json.loads((root / 'public.json').read_text())
        protocol = json.loads((root / 'protocol.json').read_text())
        self.assertEqual(n4.digest((root / 'public.json').read_bytes()), protocol['public_sha256'])
        self.assertEqual(len(data['requests']), 72)
        self.assertNotIn('gold', data['reference'][protocol['documents'][0]][0])
        for did in protocol['documents']:
            for slot in [1, 2, 3]:
                rows = [r for r in data['requests'] if r['doc_id'] == did and r['slot'] == slot]
                self.assertEqual(rows[0]['body']['messages'], rows[1]['body']['messages'])

    def test_duplicate_edges_are_one_vote_and_truncation_invalid(self):
        row = {'id_mapping': {'1': '1', '2': '2'}, 'allowed_original': [['1', '2']]}
        item = {'response': {'body': {'model': 'm', 'choices': [{'finish_reason': 'stop', 'message':
                {'content': '[{"pair":"1,2","label":"causes"},{"pair":"2,1","label":"causedby"}]'}}]}}}
        event = n4.decode_response(row, item, {'allowed_served_models': ['m']})
        self.assertTrue(event['valid'])
        self.assertEqual(len(event['positive']), 1)
        item['response']['body']['choices'][0]['finish_reason'] = 'length'
        self.assertFalse(n4.decode_response(row, item, {'allowed_served_models': ['m']})['valid'])

    def test_unknown_id_invalid_not_negative(self):
        row = {'id_mapping': {'1': '1', '2': '2'}, 'allowed_original': [['1', '2']]}
        item = {'response': {'body': {'model': 'm', 'choices': [{'finish_reason': 'stop', 'message':
                {'content': '[{"pair":"1,3","label":"causes"}]'}}]}}}
        self.assertFalse(n4.decode_response(row, item, {'allowed_served_models': ['m']})['valid'])


if __name__ == '__main__':
    unittest.main()
