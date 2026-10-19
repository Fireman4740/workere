"""Harness² ECI adapter: base-tree request parity and an offline sweep with scripted editors."""
import json
import os
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
H2 = ROOT / "experiments/jev/harness2"
DOCS = ["5_7ecbplus.xml", "5_5ecbplus.xml"]

POSTPROCESS = '''import json, sys

def main():
    data = json.load(sys.stdin)
    labels = {}
    for p in data["pairs"]:
        if p["probability"] >= 0.3:
            labels[p["source"] + "," + p["target"]] = "causal"
    print(json.dumps({"labels": labels}))

main()
'''


class BaseTreeParity(unittest.TestCase):
    def test_luna_base_tree_reproduces_frozen_request(self):
        from pipelines.harness2_eci.surface import luna_base_tree, luna_request_parts
        prompt = yaml.safe_load((ROOT / "prompts/esl_standard_eci_lean_coref.yaml").read_text())
        system, prefix, suffix = luna_request_parts(luna_base_tree(prompt), prompt)
        text = "A <1 storm> caused <2 floods> ."
        self.assertEqual(system, prompt["system"])
        self.assertEqual(prefix + "Text:\n" + text + suffix,
                         prompt["user_template"].format(doc_text=text, pair_lines="", doc_id="d"))

    def test_decision_base_tree_reproduces_frozen_question(self):
        from pipelines.harness2_eci.surface import decision_base_tree, decision_question
        from pipelines.jev.state_builder import questions_from_params
        self.assertEqual(decision_question(decision_base_tree()), questions_from_params({}))

    def test_validation_lists_every_problem(self):
        from pipelines.harness2_eci.surface import decision_base_tree, validate
        tree = {**decision_base_tree(), "plugin/x.ts": "", "scripts/other.py": "",
                "tool_descriptions/tool_guidance.md": "yes: always\n"}
        with self.assertRaises(ValueError) as caught:
            validate(tree, "decision")
        message = str(caught.exception)
        for fragment in ("plugin/x.ts", "scripts/other.py", "true:"):
            self.assertIn(fragment, message)


class OfflineSweep(unittest.TestCase):
    def test_scripted_sweep_scores_base_probes_and_selection(self):
        from benchmark.configuration import read_yaml
        tmp = Path(tempfile.mkdtemp(prefix="h2-test-"))
        template = read_yaml(H2 / "decision-executor.yaml")
        client = template["pipeline"]["relation_predictor"]["params"]["client"]
        client.update(mode="mock", cache_path=str(tmp / "cache.sqlite"))
        client["shared_budget"]["path"] = str(tmp / "budget.sqlite")
        template["execution"].update(allow_network=False, allow_paid=False)
        (H2 / "test-tmp-executor.yaml").write_text(yaml.safe_dump(template))
        self.addCleanup((H2 / "test-tmp-executor.yaml").unlink)
        os.environ["HARNESS2_IMPROVER_MODEL"] = os.environ["HARNESS2_SOLVER_MODEL"] = "openrouter/openai/gpt-6-luna"
        from harness2.core import schedule
        from harness2.core.proposer import Proposer, Reply
        from pipelines.harness2_eci.adapter import EciAdapter
        from pipelines.harness2_eci.driver import score

        class Scripted(Proposer):
            def call(self, mission, context_dir, label, **kwargs):
                context, workspace = Path(context_dir), None
                self.seen = getattr(self, "seen", set()) | {p.name for p in context.rglob("*") if p.is_file()}
                if label.startswith("experience."):
                    text = "NO_ENTRY"
                else:
                    workspace = context / "harness"
                    if ".dom" in label or label.startswith("dom") or "dom1" in label:
                        (workspace / "LongTermMEMORY.md").write_text("Arrests follow investigations.\n")
                    else:
                        (workspace / "scripts").mkdir(exist_ok=True)
                        (workspace / "scripts/postprocess.py").write_text(POSTPROCESS)
                    text = "<notes>scripted edit</notes>"
                reply = Reply(label=label, text=text, ok=True, attempts=1, workspace=workspace)
                self._calls.append(reply)
                return reply

        adapter = EciAdapter(executor="decision", template_path=H2 / "test-tmp-executor.yaml",
                             suites_dir=ROOT / "experiments/jev/round4/suites", documents=DOCS, root=tmp / "work",
                             model="openrouter/openai/gpt-6-luna")
        prop = Scripted(adapter.spec.channels["proposer"])
        sweep = schedule.run_sweep(1, adapter=adapter, prop=prop, runs_root=adapter.spec.runs_root,
                                   library_root=tmp / "work/library", modes=["agg"], seed=42, only_tasks=DOCS)
        self.assertEqual(sum(d["done"] for d in sweep["domains"]), 2, json.dumps(sweep)[:3000])
        cfg = {"documents": DOCS, "modes": ["agg"], "k": 1, "seed": 42, "executor": "decision"}
        result = score(adapter, cfg, sweep)
        self.assertEqual(result["documents_scored"], 2, json.dumps(result)[:3000])
        self.assertIn("agg:selected", result["paired_vs_base"])
        self.assertTrue({"agg:inv1", "agg:dom1", "agg:agg1"} <= set(result["micro"]))
        leaked = {n for n in prop.seen if any(k in n.lower() for k in ("metric", "gold", "annot"))}
        self.assertFalse(leaked, leaked)
        # The postprocess edit must change predictions relative to the base harness.
        a = result["micro"]["base1"]["binary"]
        b = result["micro"]["agg:inv1"]["binary"]
        self.assertNotEqual((a["tp"], a["fp"]), (b["tp"], b["fp"]))


class ProposerAccounting(unittest.TestCase):
    def test_cost_from_step_tokens_and_refusals(self):
        from pipelines.harness2_eci.proposer import session_cost
        step = {"type": "step_finish", "part": {"type": "step-finish", "cost": 0,
                "tokens": {"input": 1_000_000, "output": 100_000, "reasoning": 100_000, "cache": {"read": 0, "write": 0}}}}
        self.assertAlmostEqual(session_cost(json.dumps(step) + "\n" + json.dumps(step), (.1, .5)), .4)
        refused = {"type": "error", "error": {"name": "APIError", "data": {"statusCode": 401}}}
        self.assertEqual(session_cost(json.dumps(refused), (.1, .5)), 0.0)
        self.assertIsNone(session_cost("not json\n", (.1, .5)))

    def test_key_falls_back_to_openrouter_variable(self):
        from unittest import mock
        from pipelines.harness2_eci.proposer import openrouter_key
        with mock.patch.dict(os.environ, {"OPENROUTER": "k"}, clear=True):
            self.assertEqual(openrouter_key(), "k")
        with mock.patch.dict(os.environ, {}, clear=True), self.assertRaises(RuntimeError):
            openrouter_key()


if __name__ == "__main__":
    unittest.main()
