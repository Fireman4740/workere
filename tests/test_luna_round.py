"""Round K: Luna one-relation-per-call parsing, cascade control flow and the launch config."""
import json
import unittest
from pathlib import Path

from pipelines.judges import luna_round as k
from pipelines.judges.driver import load, plan

ROOT = Path(__file__).resolve().parents[1]


class Parsing(unittest.TestCase):
    def test_pair_decision_takes_last_object_and_checks_range(self):
        got = k.parse("pair_decision", 'think {"x": 1}\n{"causal": true, "direction": "A->B", "probability": 72}')
        self.assertEqual(got["probability"], .72)
        for bad in ('{"causal": "yes", "probability": 5}', '{"causal": true, "probability": 140}', "no json"):
            with self.assertRaises(ValueError):
                k.parse("pair_decision", bad)

    def test_type_answers_are_restricted_to_remaining_types(self):
        self.assertTrue(k.parse("type_check", '{"answer": "yes"}'))
        self.assertEqual(k.parse("type_final", '{"choice": "reaction"}', {"REACTION", "IDENTITY"}), "REACTION")
        with self.assertRaises(ValueError):
            k.parse("type_eliminate", '{"eliminate": "PREVENTION"}', {"REACTION", "IDENTITY"})

    def test_mention_lists_drop_unknown_ids(self):
        self.assertEqual(k.parse("cause_proposer", '{"causes": [3, "9", "77"]}', {"3", "9"}), ["3", "9"])
        self.assertEqual(k.parse("event_grouping", '{"groups": [["1", "2", "99"]]}', {"1", "2"}), [["1", "2"]])


class FakeClient:
    """Answers type checks from a table; records how many stages were needed."""
    def __init__(self, yes):
        self.yes, self.stages = yes, []

    def ask_many(self, requests):
        tasks = [k.task_of({"messages": [{"content": s}]}) for s, _ in requests]
        self.stages.append(tasks[0] if tasks else None)
        out = []
        for (s, u), t in zip(requests, tasks):
            if t == "type_check":
                out.append(any(q in u for q in self.yes))
            else:
                remaining = sorted(k._allowed({"messages": [{"content": s}, {"content": u}]}))
                out.append(remaining[0])
        return out


class Cascade(unittest.TestCase):
    def test_eliminates_until_two_types_then_final_choice(self):
        prompts = k.Prompts(ROOT / "prompts/esl_luna_round_k.yaml")
        yes = [prompts.types[t]["question"] for t in ("DIRECT_CAUSE", "ENABLEMENT", "REACTION", "PREVENTION")]

        class Docs:
            text = {"d": "<1 storm> caused <2 flood>"}

            @staticmethod
            def describe(doc, mentions):
                return f"mention <{mentions[0]}>"

        client = FakeClient(yes)
        out = k.cascade_one(client, prompts, Docs, "d", [("1", "2")])[("1", "2")]
        self.assertEqual(len(out["stage1_yes"]), 4)
        self.assertEqual(len(out["eliminated"]), 2)
        self.assertEqual(client.stages, ["type_check", "type_eliminate", "type_eliminate", "type_final"])
        self.assertTrue(out["causal"])

    def test_no_type_accepted_means_no_relation(self):
        prompts = k.Prompts(ROOT / "prompts/esl_luna_round_k.yaml")

        class Docs:
            text = {"d": "<1 a> <2 b>"}
            describe = staticmethod(lambda doc, m: m[0])

        out = k.cascade_one(FakeClient([]), prompts, Docs, "d", [("1", "2")])[("1", "2")]
        self.assertEqual((out["final_type"], out["causal"]), (None, False))


class LaunchConfig(unittest.TestCase):
    def test_active_config_is_luna_high_on_dev44_documents(self):
        cfg = load(ROOT / "experiments/jev/active.yaml")
        self.assertIn(cfg["experiment"], k.EXPERIMENTS)
        self.assertEqual(cfg["client"]["reasoning_effort"], "high")
        p = plan(cfg)
        self.assertEqual(len(p["documents"]), len(cfg["documents"]))
        controls = {r["doc"] for r in json.loads((ROOT / "experiments/jev/round4/controls.json").read_text())}
        self.assertTrue(set(cfg["documents"]) <= controls)


if __name__ == "__main__":
    unittest.main()
