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


class FusionExecutor(unittest.TestCase):
    def test_base_fusion_script_reproduces_frozen_rule_and_question(self):
        import random
        from pipelines.harness2_eci import postprocess
        from pipelines.harness2_eci.surface import fusion_base_tree, fusion_decision_question
        from pipelines.jev.state_builder import questions_from_params
        prompt = yaml.safe_load((ROOT / "prompts/esl_standard_eci_lean_coref.yaml").read_text())
        tree = fusion_base_tree(prompt)
        self.assertEqual(fusion_decision_question(tree), questions_from_params({}))
        rng, ids, pairs = random.Random(7), [str(i) for i in range(6)], []
        signal = {(a, b): (rng.choice([0, 1, 2, 3, 4, 5, 6, 7]) / 7, rng.random()) for a in ids for b in ids if a != b}
        for (a, b), (luna, d1) in signal.items():
            fused = ((luna + signal[(b, a)][0]) / 2 + (d1 + signal[(b, a)][1]) / 2) / 2
            pairs.append({"source": a, "target": b, "luna_fraction": luna, "luna_votes": [], "d1_probability": d1,
                          "probability": fused, "default_label": "causal" if fused >= .39 else "norel"})
        labels, log = postprocess.apply(tree, "fusion", {"document_id": "d", "executor": "fusion",
                                                         "mentions": [], "pairs": pairs})
        self.assertIn("changed 0 of", log)
        self.assertTrue(any(v == "causal" for v in labels.values()) and any(v == "norel" for v in labels.values()))

    def test_routed_proposer_walks_each_role_chain(self):
        from harness2.core.spec import ChannelSpec
        from pipelines.harness2_eci.proposer import RoutedProposer, role_of
        self.assertEqual([role_of(x) for x in ("inv2.a1", "dom1.a2", "agg2.a1", "select.agg2", "experience.agg2")],
                         ["propose", "propose", "compose", "select", "reflect"])
        cfg = read_active()
        prop = RoutedProposer(ChannelSpec("opencode", cfg["routes"]["compose"][0]), binary=Path("/bin/true"),
                              routes=cfg["routes"], models=cfg["models"])
        prop._route.label, prop._route.attempt = "inv1.a1", 0
        first, second = prop._session_model()[0], prop._session_model()[0]
        self.assertEqual([first, second], cfg["routes"]["propose"])
        prop._route.label, prop._route.attempt = "select.agg2", 0
        model, price, reserve = prop._session_model()
        self.assertEqual(model, "openrouter/deepseek/deepseek-v4.1-flash-1017")
        self.assertEqual(price, (0.15, 0.6))

    def test_unfinished_reply_is_a_failed_attempt(self):
        from unittest import mock
        from harness2.core.spec import ChannelSpec
        from pipelines.harness2_eci.proposer import OpenRouterProposer, RoutedProposer
        cfg = read_active()
        prop = RoutedProposer(ChannelSpec("opencode", cfg["routes"]["compose"][0]), binary=Path("/bin/true"),
                              routes=cfg["routes"], models=cfg["models"])
        prop._route.label = "inv1.a1"
        with mock.patch.object(OpenRouterProposer, "_opencode", return_value=("I'll start by reading MISSION.md.", 0.0, "")):
            self.assertEqual(prop._opencode(None, "", "", 1)[0], "")
        with mock.patch.object(OpenRouterProposer, "_opencode", return_value=("<notes>KEEP_BASE: no defect</notes>", 0.0, "")):
            self.assertTrue(prop._opencode(None, "", "", 1)[0])
        prop._route.label = "select.agg2"
        with mock.patch.object(OpenRouterProposer, "_opencode", return_value=("<why>x</why>\n```choice\nB\n```", 0.0, "")):
            self.assertTrue(prop._opencode(None, "", "", 1)[0])


class OfflineFusionSweep(unittest.TestCase):
    def test_k2_fusion_sweep_with_label_free_selection(self):
        from benchmark.configuration import read_yaml
        tmp = Path(tempfile.mkdtemp(prefix="h2-fusion-test-"))
        template = read_yaml(H2 / "fusion-executor.yaml")
        params = template["pipeline"]["relation_predictor"]["params"]
        for part in ("luna", "decision"):
            params[part]["client"].update(mode="mock", cache_path=str(tmp / f"{part}.sqlite"))
            params[part]["client"]["shared_budget"]["path"] = str(tmp / "budget.sqlite")
        params["luna"]["path"] = str(ROOT / "prompts/esl_standard_eci_lean_coref.yaml")
        (H2 / "test-tmp-fusion.yaml").write_text(yaml.safe_dump(template))
        self.addCleanup((H2 / "test-tmp-fusion.yaml").unlink)
        os.environ["HARNESS2_IMPROVER_MODEL"] = os.environ["HARNESS2_SOLVER_MODEL"] = "openrouter/openai/gpt-6-luna"
        from harness2.core import schedule
        from harness2.core.proposer import Proposer, Reply
        from pipelines.harness2_eci.adapter import EciAdapter
        from pipelines.harness2_eci.driver import score

        class Scripted(Proposer):
            def call(self, mission, context_dir, label, **kwargs):
                context, workspace = Path(context_dir), None
                if label.startswith("select."):
                    chosen = sorted((context / "candidates").glob("*.md"))[-1].stem
                    text = f"<why>scripted</why>\n```choice\n{chosen}\n```"
                elif label.startswith("experience."):
                    text = "NO_ENTRY"
                else:
                    workspace = context / "harness"
                    script = workspace / "scripts/postprocess.py"
                    cut = {"inv": "0.2", "dom": "0.6", "agg": "0.3"}[label[:3]]
                    script.write_text(script.read_text().replace("THRESHOLD = 0.39", f"THRESHOLD = {cut}"))
                    text = "<notes>scripted threshold</notes>"
                reply = Reply(label=label, text=text, ok=True, attempts=1, workspace=workspace)
                self._calls.append(reply)
                return reply

        adapter = EciAdapter(executor="fusion", template_path=H2 / "test-tmp-fusion.yaml",
                             suites_dir=ROOT / "experiments/jev/round4/suites", documents=DOCS, root=tmp / "work",
                             model="openrouter/openai/gpt-6-luna")
        prop = Scripted(adapter.spec.channels["proposer"])
        sweep = schedule.run_sweep(2, adapter=adapter, prop=prop, runs_root=adapter.spec.runs_root,
                                   library_root=tmp / "work/library", modes=["agg"], seed=42, only_tasks=DOCS)
        self.assertEqual(sum(d["done"] for d in sweep["domains"]), 2, json.dumps(sweep)[:3000])
        result = score(adapter, {"documents": DOCS, "modes": ["agg"], "k": 2, "seed": 42, "executor": "fusion"}, sweep)
        self.assertEqual(result["documents_scored"], 2, json.dumps(result)[:3000])
        self.assertTrue({"base1", "base2", "agg:inv2", "agg:dom2", "agg:agg2", "agg:selected"} <= set(result["micro"]))
        self.assertEqual(result["micro"]["base1"], result["micro"]["base2"])
        self.assertTrue(any(c.label.startswith("select.") for c in prop._calls))
        lo, hi = result["micro"]["agg:inv1"]["binary"], result["micro"]["agg:dom1"]["binary"]
        self.assertGreaterEqual(lo["tp"] + lo["fp"], hi["tp"] + hi["fp"])


def read_active():
    return yaml.safe_load((ROOT / "experiments/jev/active.yaml").read_text())["proposer"]


if __name__ == "__main__":
    unittest.main()
