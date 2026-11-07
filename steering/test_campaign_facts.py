"""CPU checks for source grounding, controls, and unbiased pair reuse."""
import json
import unittest

from steering import campaign_facts as f


class EventCardTests(unittest.TestCase):
    def test_accepts_only_complete_and_valid_card_json(self):
        good = {'0': {'actor': 'A', 'affected': '', 'evidence': 'A left', 'status': 'asserted'}}
        self.assertEqual(f.parse_features(json.dumps(good), 1), good)
        bad = [json.dumps({'0': {'actor': 'A'}}),
               json.dumps({'0': {**good['0'], 'status': 'invented'}}),
               '{"0":{"actor":"A","actor":"B","affected":"","evidence":"","status":"unknown"}}',
               json.dumps(good) + ' trailing text']
        for text in bad:
            self.assertIsNone(f.parse_features(text, 1), text)

    def test_unanchored_fields_are_removed_and_status_requires_evidence(self):
        source = 'On Tuesday <x arrived> the pilot met Sarah.'
        card = {'actor': 'pilot', 'affected': 'fictional person', 'evidence': 'imaginary', 'status': 'planned'}
        result, rejected = f.clean_card(card, source)
        self.assertEqual(result, {'actor': 'pilot', 'affected': '', 'evidence': '', 'status': 'unknown'})
        self.assertEqual(rejected, 2)
        card['evidence'] = 'arrived'
        result, _ = f.clean_card(card, source)
        self.assertEqual(result['evidence'], 'arrived')
        self.assertEqual(result['status'], 'planned')

    def test_permutation_preserves_cards_and_changes_identity(self):
        cards = {str(i): {'actor': str(i), 'affected': '', 'evidence': '', 'status': 'unknown'} for i in range(8)}
        shuffled = f.permute_cards(cards, 'CTB', 'doc')
        self.assertEqual(f.permute_cards(cards, 'CTB', 'doc'), shuffled)
        self.assertEqual(sorted(x['actor'] for x in cards.values()), sorted(x['actor'] for x in shuffled.values()))
        self.assertTrue(all(cards[k] != shuffled[k] for k in cards))

    def test_adding_cards_preserves_original_pair_order_and_text(self):
        original = [{'role': 'user', 'content': 'Document context:\n<1 hit> <2 fell>\n\nPairs:\n0: A=<1>, B=<2>'}]
        cards = {'1': {'actor': 'he', 'affected': '', 'evidence': 'hit', 'status': 'asserted'},
                 '2': {'actor': '', 'affected': '', 'evidence': 'fell', 'status': 'asserted'}}
        new = f.add_cards(original, cards, [('1', '2')])
        self.assertIn('Document context:\n<1 hit> <2 fell>', new[0]['content'])
        self.assertTrue(new[0]['content'].endswith('\n\nPairs:\n0: A=<1>, B=<2>'))
        self.assertNotIn('Extracted event notes', original[0]['content'])

    def test_duplicate_mentions_use_first_without_gold(self):
        result = f.event_surfaces('<1 hit> and <1 hit again> before <2 fell>', {'1','2'})
        self.assertEqual(result, {'1': 'hit', '2': 'fell'})


if __name__ == '__main__':
    unittest.main()
