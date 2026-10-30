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
import shutil
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import httpx
import yaml

from eci.config import load_config, prompt_spec
from eci.data import load_eval_docs
from eci.fewshot import FewShotBank
from eci.llm import _openrouter_chat
from eci.parse import parse_triples
from eci.runner import Runner
from eci.tracking import RunDir

SPEC = ROOT / "experiments/jev/qwen/active.yaml"


def restore_demo_caches(spec, state):
    """Reuse successful demo calls with identical model and sampling settings."""
    previous = spec.get("resume_cache_from_run")
    if not previous or state.name == "preflight" or (state / "resume-caches.json").exists():
        return
    source = Path(spec["state_root"]).expanduser() / previous
    old = yaml.safe_load((source / "transport.yaml").read_text())
    identity = ("model", "temperature", "reasoning_effort", "top_p", "top_k", "min_p",
                "presence_penalty", "repetition_penalty", "max_tokens", "chat_template_kwargs",
                "cache_namespace", "protocol_queue", "only", "corpora")
    if any(old.get(k) != spec.get(k) for k in identity):
        raise ValueError("Refusing cache reuse after a model, sampling or corpus change")
    endpoint_changed = old["base_url"] != spec["base_url"]
    if endpoint_changed and not spec.get("resume_allow_endpoint_change", False):
        raise ValueError("Endpoint change must be explicitly allowed after verifying the model service")
    receipts = []
    for name in spec["only"]:
        destination = state / name
        destination.mkdir(parents=True, exist_ok=True)
        db, output = source / name / "llm-cache.sqlite", destination / "llm-cache.sqlite"
        rows = 0
        if db.exists() and not output.exists():
            with sqlite3.connect(db.as_uri() + "?mode=ro", uri=True) as src, sqlite3.connect(output) as dst:
                src.backup(dst)
                rows = dst.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0]
        cot, cot_output = source / name / "cot-cache.json", destination / "cot-cache.json"
        if cot.exists() and not cot_output.exists():
            shutil.copy2(cot, cot_output)
        receipts.append({"corpus": name, "cached_calls_restored": rows})
    receipt = {"source_run": previous, "corpora": receipts, "evaluation_predictions_reused": False,
               "endpoint_change": {"from": old["base_url"], "to": spec["base_url"]} if endpoint_changed else None}
    (state / "resume-caches.json").write_text(json.dumps(receipt, indent=2))
    print("QWEN_CACHE_REUSE " + json.dumps(receipt), flush=True)


def restore_evaluation(spec, configs, state):
    """Copy stopped-run evidence, requiring the identical evaluation configuration."""
    previous = spec.get("resume_evaluation_from_run")
    if not previous or state.name == "preflight" or (state / "resume-evaluation.json").exists():
        return
    if previous != spec.get("resume_cache_from_run"):
        raise ValueError("Evaluation and demonstration resume sources must match")
    source = Path(spec["state_root"]).expanduser() / previous
    # restore_demo_caches checks transport identity before this function is called.
    if not (state / "resume-caches.json").exists():
        raise ValueError("Transport identity must be verified before evaluation reuse")

    def scientific_config(cfg):
        cfg = json.loads(json.dumps(cfg))
        cfg["llm_cache"].pop("path", None)
        cfg["runtime"].pop("logdir", None)
        if spec.get("resume_allow_endpoint_change", False):
            cfg["model"].pop("base_url", None)
        for key in ("cache_path", "dump_dir"):
            cfg["few_shot"]["cot_generation"].pop(key, None)
        return cfg

    receipts = []
    for name, cfg in configs.items():
        old_dir = Path((source / name / "run-dir.txt").read_text().strip())
        old_cfg = yaml.safe_load(next(old_dir.glob("*.config.yaml")).read_text())
        if scientific_config(old_cfg) != scientific_config(cfg):
            raise ValueError("Refusing evaluation reuse after configuration change: " + name)
        if cfg["experiment"]["enable_tools"] or prompt_spec(cfg).steps:
            raise ValueError("Resume requires the single-step, tool-free baseline")
        destination = state / name
        docs = old_dir / "docs.jsonl"
        completed = []
        if docs.exists():
            completed = [json.loads(line) for line in docs.read_text().splitlines()]
            completed = [row for row in completed if not row["failure"]]
            (destination / "resume-docs.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in completed))
        trace = next(old_dir.glob("*.traces.jsonl"), None)
        if trace is not None:
            shutil.copy2(trace, destination / "resume-traces.jsonl")
        else:
            (destination / "resume-traces.jsonl").write_text("")
        passes = load_resume_passes(destination / "resume-traces.jsonl")
        receipts.append({"corpus": name, "completed_documents": len(completed),
                         "valid_inference_passes": len(passes)})
    receipt = {"source_run": previous, "corpora": receipts,
               "reuse_policy": "completed documents; successful passes only on exact prompt match"}
    (state / "resume-evaluation.json").write_text(json.dumps(receipt, indent=2))
    print("QWEN_EVALUATION_RESUME " + json.dumps(receipt), flush=True)


def load_resume_passes(path):
    passes = {}
    if not path or not path.exists():
        return passes
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row["role"] != "inference" or not row["ok"]:
            continue
        if row.get("finish_reason") != "stop" or row["block"] is not None:
            continue
        parse_triples(row["messages"][-1]["kwargs"]["content"])
        key = (row["doc_id"], row["pass"], row["block"])
        if key in passes:
            raise ValueError("Duplicate valid resumed inference pass: " + str(key))
        passes[key] = row
    return passes


async def wait_for_api(spec, key, http):
    """Wait for the existing model service to finish loading before any inference."""
    deadline = time.monotonic() + spec["api_ready_timeout_seconds"]
    while True:
        try:
            response = await http.get(spec["base_url"].rstrip("/") + "/models",
                                      headers={"Authorization": "Bearer " + key}, timeout=10)
            response.raise_for_status()
            models = response.json().get("data", [])
            if any(model.get("id") == spec["model"] for model in models):
                print("QWEN_API_READY " + json.dumps({"model": spec["model"]}), flush=True)
                return
            reason = "requested model not listed"
        except (httpx.HTTPError, ValueError) as error:
            reason = type(error).__name__
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError("Model API did not become ready: " + reason)
        print("QWEN_API_WAIT " + json.dumps({"reason": reason, "remaining_seconds": round(remaining)}), flush=True)
        await asyncio.sleep(min(15, remaining))


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
                            temperature=spec["temperature"], reasoning_effort=spec["reasoning_effort"], provider_order=[])
        cfg["experiment"].update(concurrency=spec["concurrency"], doc_concurrency=corpus["documents"],
                                 timeout=spec["timeout_seconds"], retries=3, retry_wait=5)
        cfg["mlflow"].update(enabled=False, log_traces=False, log_causal_graphs=False)
        # Persistent, corpus-specific cache/checkpoint paths prevent cross-run SQLite contention.
        cfg["llm_cache"].update(path=str(state / name / "llm-cache.sqlite"), namespace=spec["cache_namespace"])
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
            reasoning_effort=spec["reasoning_effort"], top_p=spec["top_p"],
            presence_penalty=spec["presence_penalty"],
            base_url=spec["base_url"], cache=cache, max_retries=0,
            http_async_client=http, timeout=spec["timeout_seconds"],
            extra_body={"max_tokens": spec["max_tokens"], "stream": False,
                        "top_k": spec["top_k"], "min_p": spec["min_p"],
                        "repetition_penalty": spec["repetition_penalty"],
                        "chat_template_kwargs": spec["chat_template_kwargs"]})
    return make


class SharedRunner(Runner):
    def __init__(self, *args, shared_slots, resume_traces=None, resume_source=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.shared_slots = shared_slots
        self.resume_passes = load_resume_passes(resume_traces)
        self.resume_source = resume_source
        self.resumed_calls = 0

    def _clients(self):
        super()._clients()
        self.infer.slots = self.cot.slots = self.shared_slots

    async def _pass(self, msgs, meta, tools):
        key = (meta.doc_id, meta.pass_idx, meta.block)
        saved = self.resume_passes.get(key)
        if saved is not None and tools is None and not self.prompt.steps:
            original = saved["messages"][:-1]
            current = [message.to_json() for message in msgs]
            if json.dumps(original, sort_keys=True) == json.dumps(current, sort_keys=True):
                text = saved["messages"][-1]["kwargs"]["content"]
                triples = parse_triples(text)
                self.trace.write({**saved, "resumed_from_run": self.resume_source})
                self.resumed_calls += 1
                print("QWEN_PASS_RESUMED " + json.dumps({"doc_id": meta.doc_id,
                    "pass": meta.pass_idx, "source_run": self.resume_source}), flush=True)
                return text, triples, []
            print("QWEN_RESUME_PROMPT_MISMATCH " + json.dumps({"doc_id": meta.doc_id,
                "pass": meta.pass_idx}), flush=True)
        return await super()._pass(msgs, meta, tools)


async def execute(spec, configs, state, plans):
    key = credential(spec)
    count = 0

    async def observe(response):
        nonlocal count
        if response.status_code != 200 or not response.request.url.path.endswith("/chat/completions"):
            return
        await response.aread()
        payload = response.json()
        if payload.get("model") != spec["model"]:
            raise ValueError("Unexpected served model: " + str(payload.get("model")))
        count += 1
        message = payload.get("choices", [{}])[0].get("message", {})
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        print("QWEN_RESPONSE " + json.dumps({"completed_http_calls": count,
              "model": payload["model"], "usage": payload.get("usage"),
              "reasoning_characters": len(reasoning),
              "thinking_tag_in_content": "<think>" in (message.get("content") or ""),
              "finish_reason": payload.get("choices", [{}])[0].get("finish_reason")}), flush=True)

    slots = asyncio.Semaphore(spec["concurrency"])
    async with httpx.AsyncClient(
        limits=httpx.Limits(max_connections=spec["concurrency"], max_keepalive_connections=spec["concurrency"]),
        timeout=spec["timeout_seconds"], event_hooks={"response": [observe]}) as http:
        await wait_for_api(spec, key, http)
        factory = model_factory(spec, key, http)

        async def one(name, cfg):
            manifest = state / name / "run-dir.txt"
            manifest.parent.mkdir(parents=True, exist_ok=True)
            if manifest.exists():
                run_dir = RunDir(Path(manifest.read_text().strip()))
            else:
                run_dir = RunDir.create(cfg["runtime"]["logdir"])
                checkpoint = state / name / "resume-docs.jsonl"
                if checkpoint.exists():
                    shutil.copy2(checkpoint, run_dir.checkpoint)
                manifest.write_text(str(run_dir.path))
            runner = SharedRunner(cfg, run_dir, factory=factory, retry_failed=True, shared_slots=slots,
                resume_traces=state / name / "resume-traces.jsonl",
                resume_source=spec.get("resume_evaluation_from_run"))
            report = await runner.run()
            result = {"corpus": name, "run_dir": str(run_dir.path),
                      "documents": len(report["per_doc_metrics"]), "failures": report["retry_failed_docs"],
                      "resumed_partial_passes": runner.resumed_calls,
                      "resume_source_run": spec.get("resume_evaluation_from_run"),
                      "usage_scope": "new calls in this tracked run; restored responses excluded",
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
    restore_demo_caches(spec, state)
    configs = effective_configs(spec, state)
    restore_evaluation(spec, configs, state)
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
