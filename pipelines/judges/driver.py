"""Run one DeepSeek judge experiment (`workere-judge/v1`) and print its scored result."""
import json
import os
import urllib.request
from pathlib import Path

from benchmark.configuration import read_yaml
from benchmark.io import write_json

SCHEMA = "workere-judge/v1"


def _module(cfg):
    if cfg["experiment"] == "coref_agreement":
        from pipelines.judges import coref_agreement as module
    elif cfg["experiment"] == "band_verifier":
        from pipelines.judges import band_verifier as module
    else:
        raise ValueError(f"unknown judge experiment {cfg['experiment']!r}")
    return module


def load(config_path):
    cfg = read_yaml(config_path)
    if cfg.get("schema") != SCHEMA:
        raise ValueError(f"expected schema {SCHEMA}")
    base = Path(config_path).resolve().parent
    cfg["controls"], cfg["suites_dir"] = str(base / cfg["controls"]), str(base / cfg["suites_dir"])
    if cfg["client"]["reasoning_effort"] != "max":
        raise ValueError("both judge experiments require explicit reasoning effort max")
    if cfg["client"]["model"] != "deepseek/deepseek-v4.1-flash-1017":
        raise ValueError("judge model must remain the user-selected DeepSeek version")
    if cfg["workers"] < 1 or cfg["pairs_per_call"] < 1 or cfg["samples"] < 1:
        raise ValueError("workers, chunk size and samples must be positive")
    _module(cfg)
    return cfg


def plan(cfg):
    from pipelines.judges.common import Dev44
    data = Dev44(cfg["controls"], cfg["suites_dir"])
    out = {"schema": "workere-judge-plan/v1", "experiment": cfg["experiment"], "model": cfg["client"]["model"],
           "reasoning_effort": cfg["client"]["reasoning_effort"], "documents": len(data.docs),
           "samples": cfg["samples"], "budget": cfg["budget"]}
    if cfg["experiment"] == "band_verifier":
        from pipelines.judges.band_verifier import band_pairs
        out["band_unordered_pairs"] = sum(len(band_pairs(data, d, *cfg["band"])) for d in data.docs)
        out["expected_calls"] = cfg["samples"] * sum(
            (len(band_pairs(data, d, *cfg["band"])) + cfg["pairs_per_call"] - 1) // cfg["pairs_per_call"]
            for d in data.docs)
    return out


def preflight(cfg):
    """Read-only credential/model checks, before any billable call."""
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER")
    if not key:
        raise RuntimeError("missing OpenRouter credential")
    req = urllib.request.Request("https://openrouter.ai/api/v1/key", headers={"Authorization": "Bearer " + key})
    with urllib.request.urlopen(req, timeout=30) as response:
        auth = json.load(response)["data"]
    with urllib.request.urlopen("https://openrouter.ai/api/v1/models", timeout=30) as response:
        models = json.load(response)["data"]
    model = next(m for m in models if m["id"] == cfg["client"]["model"])
    if "max" not in (model.get("reasoning") or {}).get("supported_efforts", []):
        raise RuntimeError("the selected model no longer advertises effort max")
    remaining = auth.get("limit_remaining")
    if remaining is not None and remaining < cfg["client"]["max_cost_per_call_usd"] * cfg["workers"]:
        raise RuntimeError(f"OpenRouter key limit too low: {remaining:.6f} USD available")
    out = {"key_limit_remaining_usd": remaining, "model": model["id"], "requested_effort": "max",
           "supported_efforts": model["reasoning"]["supported_efforts"]}
    print("JUDGE_PREFLIGHT " + json.dumps(out), flush=True)
    return out


def run(config_path, output):
    cfg = load(config_path)
    if cfg["client"]["mode"] != "mock":
        preflight(cfg)
    print("JUDGE_PLAN " + json.dumps(plan(cfg)), flush=True)
    result = _module(cfg).run(cfg)
    result["effective_configuration"] = cfg
    from pipelines.jev.spend import SharedBudget
    with_budget = SharedBudget(**cfg["budget"])
    try:
        result["budget_status"] = with_budget.status()
    finally:
        with_budget.close()
    write_json(output, result)
    print("JUDGE_RESULT " + json.dumps(result, ensure_ascii=False), flush=True)
    return 0
