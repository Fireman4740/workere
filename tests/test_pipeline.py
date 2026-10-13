"""The whole pipeline, offline: a scripted model stands in for OpenRouter.

Covers what a unit test cannot: the orchestration (few-shot plan -> CoT pre-generation ->
resampled inference -> vote -> report), retries on every failure class, crash + resume from the
checkpoint, MLflow params logged at start, and that `experiment.concurrency` is the real
in-flight budget.

Run: python -m unittest tests.test_pipeline -v   (needs the HF datasets in the local cache)
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import httpx
import openai
from langchain_core.messages import AIMessage

from eci.config import load_config
from eci.data import load_eval_docs
from eci.runner import Runner
from eci.tracking import RunDir
from tests.fakes import Scripted, gold_answer


def small_config(tmp: Path, **extra) -> dict:
    ov = {
        "active_dataset": "event_story_line",
        "datasets": {"event_story_line": {"max_examples": 4, "prompt": "esl_standard_eci_lean"}},
        "experiment": {"resampling": {"enabled": True, "n_runs": 3}, "concurrency": 8,
                       "doc_concurrency": 4, "retries": 2, "timeout": 30},
        "few_shot": {"enabled": True, "n_examples": 1, "pool_size": 20,
                     "cot_generation": {"cache_path": str(tmp / "cot.json"), "dump_dir": str(tmp / "dumps")}},
        "llm_cache": {"path": str(tmp / "llm_cache.sqlite")},
        "mlflow": {"enabled": True, "tracking_uri": f"sqlite:///{tmp / 'mlflow.db'}", "log_traces": False},
        "runtime": {"logdir": str(tmp / "runs")},
    }
    for k, v in extra.items():
        ov[k] = {**ov.get(k, {}), **v} if isinstance(v, dict) else v
    return load_config(overrides=ov)


def oracle(cfg):
    docs = load_eval_docs(cfg)
    return docs, gold_answer({d.text: d for d in docs})


class Pipeline(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def run_once(self, cfg, model, run_dir=None, retry_failed=False):
        run_dir = run_dir or RunDir.create(cfg["runtime"]["logdir"])
        runner = Runner(cfg, run_dir, factory=lambda *a: model, retry_failed=retry_failed)
        return asyncio.run(runner.run()), run_dir

    def test_oracle_run_scores_perfectly_and_is_tracked(self):
        cfg = small_config(self.tmp)
        docs, respond = oracle(cfg)
        model = Scripted(respond=respond)
        report, run_dir = self.run_once(cfg, model)
        self.assertEqual(report["binary"]["f1"], 1.0)
        self.assertEqual(report["retry_failed_docs"], 0)
        self.assertEqual(len(run_dir.load_docs()), 4)
        # 4 docs x 3 passes, plus 2 synthesis calls (guided + decontaminate) per planned demo
        n_demos = len(json.loads((self.tmp / "cot.json").read_text()))
        self.assertEqual(report["usage"]["roles"]["inference"]["calls"], 12)
        self.assertEqual(report["usage"]["roles"]["cot"]["calls"], 2 * n_demos)
        self.assertAlmostEqual(report["usage"]["cost_usd"], 0.001 * (12 + 2 * n_demos), places=6)
        traces = run_dir.file(".traces.jsonl").read_text().splitlines()
        self.assertEqual(len(traces), 12 + 2 * n_demos)
        self.assertTrue(all(json.loads(t)["doc_id"] for t in traces))
        # MLflow: params and final metrics on one FINISHED run
        con = sqlite3.connect(self.tmp / "mlflow.db")
        (status,) = con.execute("select status from runs").fetchone()
        n_params = con.execute("select count(*) from params").fetchone()[0]
        f1 = con.execute("select value from latest_metrics where key='binary_f1'").fetchone()[0]
        self.assertEqual(status, "FINISHED")
        self.assertGreater(n_params, 50)
        self.assertEqual(f1, 1.0)

        # A second identical run reuses every CoT: no synthesis call at all.
        model2 = Scripted(respond=respond)
        report2, _ = self.run_once(cfg, model2)
        self.assertNotIn("cot", report2["usage"]["roles"])
        self.assertEqual(model2.calls, 12)

    def test_demo_plan_is_reproducible_and_never_self(self):
        cfg = small_config(self.tmp, datasets={"event_story_line": {"max_examples": 30}})
        from eci.config import prompt_spec
        from eci.fewshot import FewShotBank
        docs = load_eval_docs(cfg)
        plans = []
        for _ in range(2):
            bank = FewShotBank(cfg, prompt_spec(cfg))
            bank.load(lambda d: None)
            bank.build_plan(docs)
            plans.append({k: [[d.id for d in s] for s in v] for k, v in bank.plan.items()})
            for doc in docs:
                for s in bank.plan[doc.id]:
                    for d in s:
                        self.assertNotEqual(d.id, doc.id)
                        self.assertNotEqual(d.idx % 5, doc.idx % 5)   # ESC: 5-fold exclusion
        self.assertEqual(plans[0], plans[1])

    def test_every_failure_class_is_retried(self):
        cfg = small_config(self.tmp, few_shot={"enabled": False})
        docs, respond = oracle(cfg)
        target = docs[0].text
        seen = {"n": 0}
        req = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
        failures = [
            "I will not answer in JSON.",                                        # parse failure
            AIMessage(content='[{"pair": "1,2"', response_metadata={"finish_reason": "length"}),  # truncated
            openai.APIConnectionError(request=req),                              # transport error
        ]

        def flaky(messages, idx):
            if target in messages[-1].content and seen["n"] < len(failures):
                seen["n"] += 1
                return failures[seen["n"] - 1]
            return respond(messages, idx)

        report, run_dir = self.run_once(cfg, Scripted(respond=flaky))
        self.assertEqual(report["retry_failed_docs"], 0)
        self.assertEqual(report["binary"]["f1"], 1.0)
        self.assertEqual(report["usage"]["roles"]["inference"]["failed_attempts"], 3)
        errors = [json.loads(l)["error"] for l in run_dir.file(".traces.jsonl").read_text().splitlines()]
        self.assertEqual(sum(e is not None for e in errors), 3)

    def test_every_attempt_is_a_full_mlflow_trace(self):
        """One trace per attempt of every role, failed and timed-out ones included, holding the
        whole conversation sent and every field of each answer. Before: inference only, the last
        user turn and the answer text only, and a timed-out attempt's trace was never written
        (its autologged span stayed open, and MLflow writes a trace once all its spans end)."""
        import mlflow
        cfg = small_config(self.tmp, mlflow={"log_traces": True}, experiment={"timeout": 1})
        docs, respond = oracle(cfg)
        slow, broken = docs[0].text, docs[1].text
        seen = set()

        def answer(messages, idx):
            if broken in messages[-1].content and "broken" not in seen:
                seen.add("broken")
                return "I will not answer in JSON."
            return AIMessage(content=respond(messages, idx), additional_kwargs={"reasoning": f"thought {idx}"},
                             response_metadata={"finish_reason": "stop", "provider": "Novita"},
                             usage_metadata={"input_tokens": 100, "output_tokens": 10, "total_tokens": 110})

        class Model(Scripted):
            async def _agenerate(self, messages, stop=None, run_manager=None, **kw):
                if slow in messages[-1].content and "slow" not in seen:
                    seen.add("slow")
                    await asyncio.sleep(3)
                return await super()._agenerate(messages, stop, run_manager, **kw)

        report, run_dir = self.run_once(cfg, Model(respond=answer))
        self.assertEqual(report["binary"]["f1"], 1.0)
        mlflow.flush_trace_async_logging()
        attempts = [json.loads(l) for l in run_dir.file(".traces.jsonl").read_text().splitlines()]
        exp = mlflow.get_experiment_by_name(cfg["mlflow"]["experiment_name"]).experiment_id
        traces = mlflow.search_traces(locations=[exp], return_type="list", max_results=10_000)
        self.assertEqual(len(traces), len(attempts))
        self.assertEqual({t.info.tags["role"] for t in traces}, {"inference", "cot"})
        self.assertEqual(sorted(t.info.state.value for t in traces if t.info.state.value != "OK"), ["ERROR"] * 2)
        roots = [next(s for s in t.data.spans if s.parent_id is None) for t in traces]
        # Every message of every attempt, as the trace file has them.
        self.assertEqual(sum(len(r.inputs["messages"]) + len(r.outputs["messages"]) for r in roots),
                         sum(len(a["messages"]) for a in attempts))
        for t, root in zip(traces, roots):
            if t.info.tags["role"] == "inference":
                self.assertEqual([m["role"] for m in root.inputs["messages"]],
                                 ["system", "user", "assistant", "user"])          # one demo
            if t.info.state.value == "OK":
                out = root.outputs["messages"][-1]
                self.assertTrue(out["reasoning"].startswith("thought"))
                self.assertEqual(out["response_metadata"]["provider"], "Novita")
                self.assertEqual(t.info.tags["provider"], "Novita")
            self.assertTrue(any(s.span_type == "CHAT_MODEL" and s.parent_id for s in t.data.spans))  # autologged
        timed_out = next(t for t in traces if "TimeoutError" in (t.data.spans[0].status.description or ""))
        self.assertTrue(all(s.status.status_code == "ERROR" for s in timed_out.data.spans))
        # The serving provider, per run: every answer that arrived except the unparseable one.
        served = report["usage"]["total"]["calls"] - 1
        self.assertEqual(report["usage"]["providers"], {"Novita": served})
        run = mlflow.MlflowClient().get_run((run_dir.path / "mlflow_run_id").read_text())
        self.assertEqual((run.data.tags["provider"], run.data.tags["provider_calls"]), ("Novita", f"Novita:{served}"))

    def test_crash_then_resume_pays_only_for_the_rest(self):
        cfg = small_config(self.tmp, few_shot={"enabled": False},
                           experiment={"doc_concurrency": 1, "concurrency": 3})
        docs, respond = oracle(cfg)
        crash_on = docs[2].text

        def crashing(messages, idx):
            if crash_on in messages[-1].content:
                raise KeyboardInterrupt  # the process dies mid-run
            return respond(messages, idx)

        run_dir = RunDir.create(cfg["runtime"]["logdir"])
        with self.assertRaises(KeyboardInterrupt):
            self.run_once(cfg, Scripted(respond=crashing), run_dir)
        finished = run_dir.load_docs()
        self.assertEqual(len(finished), 2)
        con = sqlite3.connect(self.tmp / "mlflow.db")
        self.assertGreater(con.execute("select count(*) from params").fetchone()[0], 50)  # params survived

        model = Scripted(respond=respond)
        report, _ = self.run_once(run_dir.read_config(), model, run_dir)
        self.assertEqual(model.calls, 2 * 3)                 # only the two unfinished documents
        self.assertEqual(len(report["per_doc_metrics"]), 4)
        self.assertEqual(report["binary"]["f1"], 1.0)
        runs = con.execute("select status from runs").fetchall()
        self.assertEqual(runs, [("FINISHED",)])              # the same MLflow run, reopened

    def test_concurrency_is_the_real_in_flight_budget(self):
        cfg = small_config(self.tmp, few_shot={"enabled": False},
                           datasets={"event_story_line": {"max_examples": 40}},
                           experiment={"concurrency": 60, "doc_concurrency": 40},
                           mlflow={"enabled": False})
        docs, respond = oracle(cfg)
        model = Scripted(respond=respond, delay=0.3)
        self.run_once(cfg, model)
        # 40 docs x 3 passes = 120 calls. agentere's sync graph node capped this at 28.
        self.assertEqual(model.peak, 60)


if __name__ == "__main__":
    unittest.main()


class Features(unittest.TestCase):
    """Each capability switch, end to end on a scripted oracle."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def run_cfg(self, cfg, respond=None, label_filter=("causes", "causedby")):
        docs = load_eval_docs(cfg)
        model = Scripted(respond=respond or gold_answer({d.text: d for d in docs}, label_filter))
        runner = Runner(cfg, RunDir.create(cfg["runtime"]["logdir"]), factory=lambda *a: model)
        return asyncio.run(runner.run()), model, runner

    def test_mention_split_partitions_and_logs_leakage(self):
        cfg = small_config(self.tmp, few_shot={"enabled": False}, mention_split={"enabled": True},
                           mlflow={"enabled": False})
        report, model, _ = self.run_cfg(cfg)
        self.assertEqual(model.calls, 4 * 3 * 3)                   # docs x passes x blocks
        self.assertEqual(report["binary"]["f1"], 1.0)              # out-of-block copies dropped
        leak = report["mention_split_leakage"]["overall"]
        self.assertGreater(leak["out_of_block"], 0)
        self.assertEqual(leak["in_block"] / 3, report["binary"]["support_pos"])

    def test_oracle_whitelist_enforced_and_gold_count(self):
        cfg = small_config(self.tmp, few_shot={"enabled": False}, mlflow={"enabled": False},
                           mention_whitelist={"enabled": True, "mode": "oracle", "enforce": True},
                           gold_count={"enabled": True})
        report, _, _ = self.run_cfg(cfg)
        self.assertEqual(report["mention_whitelist"]["out_of_scope"], 0)
        self.assertEqual(report["binary"]["f1"], 1.0)

    def test_vote_to_binary_and_binary_undirected(self):
        cfg = small_config(self.tmp, few_shot={"enabled": False}, mlflow={"enabled": False},
                           data={"vote_to_binary": True})
        report, _, _ = self.run_cfg(cfg)
        self.assertEqual(report["binary"]["f1"], 1.0)
        self.assertEqual(set(report["per_label"]) - {"norel"}, {"causal"})
        cfg = small_config(self.tmp, few_shot={"enabled": False}, mlflow={"enabled": False},
                           data={"binary_undirected": True})
        report, _, _ = self.run_cfg(cfg, label_filter=None)
        self.assertEqual(report["binary"]["f1"], 1.0)

    def test_multi_step_prompt_with_multi_step_cot(self):
        cfg = small_config(self.tmp, active_dataset="meci", mlflow={"enabled": False},
                           datasets={"meci": {"max_examples": 2, "prompt": "meci_standard_eci_steps"}},
                           experiment={"resampling": {"enabled": False, "n_runs": 1}})
        report, model, runner = self.run_cfg(cfg)
        steps = len(runner.prompt.steps)
        self.assertEqual(report["usage"]["roles"]["inference"]["calls"], 2 * (steps + 1))
        # synthesis: one guided call per turn, then one decontamination per turn
        self.assertEqual(report["usage"]["roles"]["cot"]["calls"] % (2 * (steps + 1)), 0)
        self.assertEqual(report["binary"]["f1"], 1.0)

    def test_similarity_density_and_human_rationales(self):
        cfg = small_config(self.tmp, mlflow={"enabled": False},
                           few_shot={"selection": "similarity", "density_stratify": True, "n_examples": 2})
        report, _, runner = self.run_cfg(cfg)
        self.assertEqual(report["binary"]["f1"], 1.0)
        cfg = small_config(self.tmp, mlflow={"enabled": False},
                           few_shot={"human_rationales": {"dir": "few_shot_rationales", "only": True},
                                     "n_examples": 1},
                           experiment={"resampling": {"enabled": False, "n_runs": 1}})
        report, _, runner = self.run_cfg(cfg)
        self.assertEqual({d.id for d in runner.bank.pool}, {"1_15ecbplus.xml", "4_1ecbplus.xml"})
        self.assertNotIn("cot", report["usage"]["roles"])        # rationales are never synthesized

    def test_tool_loop(self):
        cfg = small_config(self.tmp, few_shot={"enabled": False}, mlflow={"enabled": False},
                           experiment={"enable_tools": True, "tools": ["coherence", "reprompt", "placeholder"],
                                       "resampling": {"enabled": False, "n_runs": 1}})
        docs = load_eval_docs(cfg)
        oracle_fn = gold_answer({d.text: d for d in docs})

        def with_tools(messages, idx):
            if not any(m.type == "tool" for m in messages):
                return AIMessage(content="", tool_calls=[
                    {"id": "c1", "name": "coherence", "args": {"pairs": [{"pair": "1,2", "label": "causes"}]}},
                    {"id": "c2", "name": "reprompt", "args": {}}],
                    response_metadata={"finish_reason": "tool_calls"})
            return oracle_fn(messages, idx)

        report, model, runner = self.run_cfg(cfg, with_tools)
        self.assertEqual(report["binary"]["f1"], 1.0)
        self.assertEqual(model.calls, 4 * 2)                      # one tool round, then the answer
        tool_msgs = [m for line in runner.run_dir.file(".traces.jsonl").read_text().splitlines()
                     for m in json.loads(line)["messages"] if m["id"][-1] == "ToolMessage"]
        self.assertEqual(len(tool_msgs), 8)
        self.assertTrue(any("incoherence violations" in m["kwargs"]["content"] for m in tool_msgs))
