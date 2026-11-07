"""CPU tests for pair coverage, strict scoring inputs, and safe resume."""
import json
from pathlib import Path
import tempfile
import unittest

from steering import campaign_blocks as c


class CampaignTests(unittest.TestCase):
    def test_strict_final_answer(self):
        self.assertEqual(c.parse_answer('{"0":"A","1":"N"}', 2), {"0": "A", "1": "N"})
        for text in ('{"0":"A"}', '{"0":"A","1":"B","2":"N"}',
                     '{"0":"A","0":"N","1":"B"}', 'reason {"0":"A","1":"B"}',
                     '{"0":"A","1":"unknown"}', '{"0":"A","1":'):
            self.assertIsNone(c.parse_answer(text, 2), text)

    def test_windows_keep_each_mention_and_skip_distant_content(self):
        text = 'One. Two <x arrived>. Three. Four. Five. Six <y left>. Seven. Eight. Nine.'
        context = c.windows(text, {"x", "y"})
        self.assertIn('<x arrived>', context)
        self.assertIn('<y left>', context)
        self.assertNotIn('Nine.', context)
        with self.assertRaises(ValueError):
            c.windows(text, {"missing"})

    def test_selection_is_label_independent(self):
        ctb, reference, esc, _ = c.inputs()
        first = c.select_docs(ctb, reference, list(esc.values()))
        for doc in list(ctb.values()) + list(esc.values()):
            doc["gold"] = []
        second = c.select_docs(ctb, reference, list(esc.values()))
        self.assertEqual(first, second)
        self.assertEqual((len(first["CTB"]), len(first["ESC"])), (20, 12))
        self.assertEqual(len({doc.split('_')[0] for doc in first['ESC']}), 12)

    def test_f1_keeps_unanswered_positive_as_false_negative(self):
        universe = {("1", "2"), ("2", "1")}
        gold = set(universe)
        counts = c.counts(set(), gold, universe)
        self.assertEqual(counts, {"tp": 0, "fp": 0, "fn": 2})
        self.assertEqual(c.metric({"doc": counts})['f1'], 0)

    def test_resume_cache_rejects_foreign_and_duplicate_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'gen.jsonl'
            path.write_text('{"key":"a"}\n')
            self.assertEqual(c.load_completed(path, {"a", "b"}), {"a"})
            with self.assertRaises(ValueError):
                c.load_completed(path, {"b"})
            path.write_text('{"key":"a"}\n{"key":"a"}\n')
            with self.assertRaises(ValueError):
                c.load_completed(path, {"a"})

    def test_restart_recovers_only_the_incomplete_last_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'gen.jsonl'
            path.write_text('{"key":"a"}\n{"key":')
            self.assertEqual(c.load_completed(path, {"a", "b"}), {"a"})
            self.assertEqual(path.read_text(), '{"key":"a"}\n')
            path.write_text('{"key":"a"}')
            self.assertEqual(c.load_completed(path, {"a"}), {"a"})
            self.assertTrue(path.read_text().endswith('\n'))


if __name__ == '__main__':
    unittest.main()
