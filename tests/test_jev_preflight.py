import os
import tempfile
import unittest
from pathlib import Path

from benchmark.io import read_json
from pipelines.jev.client import DecisionClient
from pipelines.jev.preflight import check_multi_questions


class MultiQuestionPrerequisite(unittest.TestCase):
    def test_distinct_answers_and_causal_invariance_are_required(self):
        suite_path = str(Path("experiments/jev/frozen/seen-smoke-suite.json").resolve())
        questions = {key: {"type": "noul", "instructions": key,
                          "criteria": {"true": "yes", "false": "no"}}
                     for key in ("causal", "same_event")}
        cwd = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as temp:
                os.chdir(temp)
                for alters_causal in (False, True):
                    def transport(body):
                        return {"model": "mock/test", "usage": {"cost": 0}, "answers": {
                            key: {"type": "noul", "noul": (.41 if alters_causal and len(body["questions"]) > 1 else .4)
                                  if key == "causal" else .8} for key in body["questions"]}}
                    with DecisionClient(model="test", model_version="test", transport=transport,
                            cache_path=str(Path(temp) / f"{alters_causal}.sqlite"), max_calls=1000) as client:
                        if alters_causal:
                            with self.assertRaisesRegex(ValueError, "prerequisite failed"):
                                check_multi_questions(client, questions, suite_path, 8)
                        else:
                            result = check_multi_questions(client, questions, suite_path, 8)
                            self.assertTrue(result["passed"])
                            self.assertEqual(result["pairs"], 156)
                            self.assertEqual(result["max_causal_probability_delta"], 0)
        finally:
            os.chdir(cwd)
