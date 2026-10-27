"""Tests unitaires sans modèles, sans réseau, sans jeux ESC."""
import json
import tempfile
import unittest
from pathlib import Path

import pilot


class ProtocolTests(unittest.TestCase):
    def test_canonical_nondirectional(self):
        self.assertEqual(pilot.canon('x','y'), pilot.canon('y','x'))
        with self.assertRaises(ValueError):
            pilot.canon('x','x')

    def test_gold_fields_rejected(self):
        with self.assertRaises(ValueError):
            pilot.check_no_gold({'pairs':[{'a':'e1','b':'e2','gold_label':1}]})

    def test_preserve_mentions_and_offsets(self):
        text='La pluie a provoqué un retard. Un feu a causé une fuite.'
        ms=[{'id':'e1','start':3,'end':8,'surface':'pluie'},
            {'id':'e2','start':26,'end':32,'surface':'retard'},
            {'id':'e3','start':37,'end':40,'surface':'feu'},
            {'id':'e4','start':49,'end':54,'surface':'fuite'}]
        # Use true spans to avoid a separate fixture indexing error.
        ms=[dict(id=m['id'],start=text.index(m['surface']),end=text.index(m['surface'])+len(m['surface']),surface=m['surface']) for m in ms]
        start=text.index('a provoqué')
        transformed, updated=pilot.apply_edits(text,ms,[{'start':start,'end':start+len('a provoqué'),'before':'a provoqué','after':'a précédé'}])
        self.assertIn('a précédé',transformed)
        self.assertTrue(all(transformed[m['start']:m['end']]==m['surface'] for m in updated))

    def test_do_not_edit_mentions(self):
        text='Le feu a causé une fuite.'
        mentions=[{'id':'e1','start':3,'end':6,'surface':'feu'}]
        with self.assertRaises(ValueError):
            pilot.apply_edits(text,mentions,[{'start':3,'end':6,'before':'feu','after':'incendie'}])

    def test_replace_only_if_before_matches(self):
        with self.assertRaises(ValueError):
            pilot.apply_edits('abc',[],[{'start':0,'end':1,'before':'z','after':'x'}])

    def test_score_contrast_algebra(self):
        # If O=1, T=0, P=(1,1,1), then C=1. O cancels algebraically.
        original, target, placebo=[1], [0], [1,1,1]
        c=(original[0]-target[0])-__import__('statistics').median([original[0]-x for x in placebo])
        self.assertEqual(c, __import__('statistics').median(placebo)-target[0])

    def test_selection_is_hash_stable(self):
        self.assertEqual(pilot.rank_hash('a','one',3),pilot.rank_hash('a','one',3))
        self.assertNotEqual(pilot.rank_hash('a','one',3),pilot.rank_hash('b','one',3))

    def test_f1_threshold(self):
        TP,FP,FN=464,473,507
        f0=2*TP/(2*TP+FP+FN)
        self.assertAlmostEqual(TP/(2*TP+FP+FN), f0/2)
        self.assertGreater(2*(TP+35)/(2*TP+FP+FN+100),f0)


if __name__=='__main__':
    unittest.main()
