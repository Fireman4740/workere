"""Qwen-only evaluation with the historical one-shot CoT, seven-vote protocol.

The fixed project entry launches this controller on the box. --plan performs
dataset/demo validation without any model requests. Both datasets share one
HTTP concurrency limit; credentials stay on the execution host.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import httpx
import yaml

from eci.config import load_config, prompt_spec
from eci.data import load_eval_docs
from eci.fewshot import FewShotBank
from eci.llm import _openrouter_chat
from eci.runner import Runner
from eci.tracking import RunDir

SPEC = ROOT / "experiments/jev/qwen/active.yaml"


def credential(spec):
    """Read only the selected key; never source arbitrary shell configuration."""
    name = spec["credential_name"]
    if os.environ.get(name):
        return os.environ[name]
    for line in Path(spec["credential_file"]).expanduser().read_text().splitlines():
        key, separator, raw = line.strip().removeprefix("export ").partition("=")
        if separator and key.strip() == name:
            values = shlex.split(raw, comments=True)
            if len(values) == 1 and values[0]:
                return values[0]
    raise RuntimeError(f"Missing {name} on execution host")


def effective_configs(spec, state):
    queue = yaml.safe_load((ROOT / spec["protocol_queue"]).read_text())
    entries = {entry["name"]: entry for entry in queue["experiments"]}
    configs = {}
    for name in spec["only"]:
        cfg = load_config(overrides=entries[name]["overrides"])
        corpus = spec["corpora"][name]
        assert cfg["active_dataset"] == corpus["key"]
        cfg["datasets"][corpus["key"]]["revision"] = corpus["revision"]
        cfg["model"].update(default_model_id=spec["model"], base_url=spec["base_url"],
                            temperature=spec["temperature"], reasoning_effort=None, provider_order=[])
        cfg["experiment"].update(concurrency=spec["concurrency"], doc_concurrency=corpus["documents"],
                                 timeout=spec["timeout_seconds"], retries=3, retry_wait=5)
        cfg["mlflow"].update(enabled=False, log_traces=False, log_causal_graphs=False)
        # Persistent, corpus-specific cache/checkpoint paths prevent cross-run SQLite contention.
        cfg["llm_cache"].update(path=str(state / name / "llm-cache.sqlite"), namespace="qwen38-nothink-t07-32k")
        cfg["few_shot"]["cot_generation"].update(
            cache_path=str(state / name / "cot-cache.json"), dump_dir=str(state / name / "cot-dumps"))
        cfg["runtime"]["logdir"] = str(state / name / "runs")
        configs[name] = cfg
    return configs


def validate_plan(spec, configs, state):
    plans = {}
    for name, cfg in configs.items():
        docs = load_eval_docs(cfg)
        corpus = spec["corpora"][name]
        assert len(docs) == corpus["documents"], (name, len(docs))
        pairs = sum(len(doc.pair_list) for doc in docs)
        assert pairs == corpus["candidate_pairs"], (name, pairs)
        bank = FewShotBank(cfg, prompt_spec(cfg))
        bank.load(lambda docs: None)  # Syntax and oracle coreference are disabled.
        bank.build_plan(docs)
        bank.load_cache()
        for doc in docs:
            assert len(bank.plan[doc.id]) == 7
            for group in bank.plan[doc.id]:
                assert len(group) == 1
                assert group[0].id != doc.id
                folds = cfg["datasets"][cfg["active_dataset"]]["kfold"]["n_folds"]
                assert group[0].idx % folds != doc.idx % folds
        plans[name] = {"documents": len(docs), "candidate_pairs": pairs,
                       "revision": corpus["revision"], "inference_calls": len(docs) * 7,
                       "missing_cot_documents": len(bank.missing_cots()),
                       "demo_pool": [d.id for d in bank.pool],
                       "prompt": prompt_spec(cfg).name, "folds": folds}
        print("QWEN_PLAN " + json.dumps({name: plans[name]}), flush=True)
    state.mkdir(parents=True, exist_ok=True)
    (state / "plan.json").write_text(json.dumps(plans, indent=2))
    return plans


def model_factory(spec, key, http):
    def make(model_id, temperature, reasoning_effort, cache):
        if model_id != spec["model"]:
            # Runner constructs dormant tool clients too. They must use the same
            # local model; tools are disabled and cannot issue outside calls.
            model_id = spec["model"]
        return _openrouter_chat()(
            model=model_id, temperature=spec["temperature"], api_key=key,
            base_url=spec["base_url"], cache=cache, max_retries=0,
            http_async_client=http, timeout=spec["timeout_seconds"],
            extra_body={"max_tokens": spec["max_tokens"], "stream": False,
                        "chat_template_kwargs": spec["chat_template_kwargs"]})
    return make


class SharedRunner(Runner):
    def __init__(self, *args, shared_slots, **kwargs):
        super().__init__(*args, **kwargs)
        self.shared_slots = shared_slots

    def _clients(self):
        super()._clients()
        self.infer.slots = self.cot.slots = self.shared_slots


async def execute(spec, configs, state, plans):
    key = credential(spec)
    count = 0

    async def observe(response):
        nonlocal count
        if response.status_code != 200:
            return
        await response.aread()
        payload = response.json()
        if payload.get("model") != spec["model"]:
            raise ValueError("Unexpected served model: " + str(payload.get("model")))
        count += 1
        print("QWEN_RESPONSE " + json.dumps({"completed_http_calls": count,
              "model": payload["model"], "usage": payload.get("usage"),
              "finish_reason": payload.get("choices", [{}])[0].get("finish_reason")}), flush=True)

    slots = asyncio.Semaphore(spec["concurrency"])
    async with httpx.AsyncClient(
        limits=httpx.Limits(max_connections=spec["concurrency"], max_keepalive_connections=spec["concurrency"]),
        timeout=spec["timeout_seconds"], event_hooks={"response": [observe]}) as http:
        factory = model_factory(spec, key, http)

        async def one(name, cfg):
            manifest = state / name / "run-dir.txt"
            manifest.parent.mkdir(parents=True, exist_ok=True)
            if manifest.exists():
                run_dir = RunDir(Path(manifest.read_text().strip()))
            else:
                run_dir = RunDir.create(cfg["runtime"]["logdir"])
                manifest.write_text(str(run_dir.path))
            runner = SharedRunner(cfg, run_dir, factory=factory, retry_failed=True, shared_slots=slots)
            report = await runner.run()
            result = {"corpus": name, "run_dir": str(run_dir.path),
                      "documents": len(report["per_doc_metrics"]), "failures": report["retry_failed_docs"],
                      **{k: report[k] for k in ("binary", "binary_intra", "binary_inter", "usage")}}
            complete = result["documents"] == plans[name]["documents"] and not result["failures"]
            result["complete"] = complete
            (state / name / "summary.json").write_text(json.dumps(result, indent=2))
            print(("QWEN_RESULT " if complete else "QWEN_INCOMPLETE ") + json.dumps(result), flush=True)
            if not complete:
                raise RuntimeError(f"Incomplete corpus: {name}")
            return result

        outcomes = await asyncio.gather(*(one(name, cfg) for name, cfg in configs.items()), return_exceptions=True)
        errors = [str(result) for result in outcomes if isinstance(result, BaseException)]
        if errors:
            print("QWEN_ERRORS " + json.dumps(errors), flush=True)
            return 1
        (state / "summary.json").write_text(json.dumps(outcomes, indent=2))
        print("QWEN_COMPLETE " + json.dumps({"corpora": len(outcomes), "http_calls": count}), flush=True)
        return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    spec = yaml.safe_load(SPEC.read_text())
    run_id = os.environ.get("ORX_RUN_ID", "preflight")
    state = Path(spec["state_root"]).expanduser() / run_id
    configs = effective_configs(spec, state)
    print("QWEN_EFFECTIVE_CONFIG " + json.dumps({**spec, "run_id": run_id, "state": str(state),
          "decision_model": False, "few_shot": "1 gold-guided CoT demo/pass; rewrite; same-fold exclusion",
          "passes": 7, "vote": "plurality with implicit norel", "corpora_parallel": True}), flush=True)
    plans = validate_plan(spec, configs, state)
    (state / "transport.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    if args.plan:
        return 0
    return asyncio.run(execute(spec, configs, state, plans))


if __name__ == "__main__":
    raise SystemExit(main())
