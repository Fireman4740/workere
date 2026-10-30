import unittest
from main import bound, replay


class StoppingTests(unittest.TestCase):
    def test_extrema_cover_every_remaining_vote_count(self):
        for k in range(8):
            for a in range(k+1):
                for b in range(k+1):
                    for d in (0, .2, .39, .5, .78, 1):
                        fixed = bound(a, b, d, d, k)
                        for x in range(8-k):
                            for y in range(8-k):
                                final = ((a+x)/7+(b+y)/7+2*d)/4 >= .39
                                if fixed is not None:
                                    self.assertEqual(fixed, final)

    def document(self, positive, d):
        return {"id": "fixture", "pairs": [["a", "b"], ["b", "a"]],
                "d1": [["a", "b", d], ["b", "a", d]],
                "positive_pairs_by_slot": [[["a", "b"], ["b", "a"]] if positive else [] for _ in range(7)],
                "response_ids": [str(i) for i in range(7)],
                "usages": [{"prompt_tokens": 10, "completion_tokens": 10} for _ in range(7)]}

    def test_all_negative_can_stop_after_two(self):
        self.assertEqual(replay(self.document(False, 0))["stop_after"], 2)

    def test_fixed_d1_can_stop_without_luna(self):
        self.assertEqual(replay(self.document(False, .9))["stop_after"], 0)

    def test_target_gold_is_rejected(self):
        d = self.document(False, 0)
        d["gold"] = []
        with self.assertRaises(AssertionError):
            replay(d)


if __name__ == "__main__":
    unittest.main()
