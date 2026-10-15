"""Strict experiment YAML, relative paths, seed repetitions, campaigns and Cartesian sweeps."""
from __future__ import annotations

import copy
import itertools
import re
from dataclasses import dataclass
from pathlib import Path

from benchmark.io import read_json, read_jsonl
from benchmark.schema import PROTOCOLS
from benchmark.statistics import validate_bootstrap
from benchmark.suite import documents, freeze

ALIASES = {"ctb": "causal_timebank", "esc": "event_story_line", "maven": "maven_ere"}


@dataclass(frozen=True)
class ExperimentConfig:
    data: dict
    source: Path


def read_yaml(path):
    import yaml
    with Path(path).open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return value


def _keys(value, allowed, required, name):
    if not isinstance(value, dict) or set(value) - set(allowed) or not set(required) <= set(value):
        raise ValueError(f"invalid {name} fields; required={list(required)}, allowed={list(allowed)}")


def _integer(value, name):
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")


def resolve_config(value, source):
    cfg, source = copy.deepcopy(value), Path(source).resolve()
    _keys(cfg, ("schema", "name", "dataset", "protocol", "pipeline", "plugins", "models", "seed", "seeds",
                "evaluation", "execution", "output_dir"),
          ("schema", "name", "dataset", "protocol", "pipeline", "models"), "experiment")
    if cfg["schema"] != "workere-experiment/v1":
        raise ValueError("unsupported experiment schema")
    if cfg["protocol"] == "relation_gold_mentions":
        cfg["protocol"] = "relations_gold"
    if not isinstance(cfg["name"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", cfg["name"]):
        raise ValueError("experiment name must be a simple nonempty slug")
    if cfg["protocol"] not in PROTOCOLS:
        raise ValueError("unsupported experiment protocol")
    if "seed" in cfg and "seeds" in cfg:
        raise ValueError("choose seed or seeds, not both")
    seeds = cfg.get("seeds", [cfg.get("seed", 42)])
    if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a nonempty array without duplicates")
    for seed in seeds:
        _integer(seed, "seed")
    if "seeds" not in cfg:
        cfg["seed"] = seeds[0]
    ds = cfg["dataset"]
    _keys(ds, ("name", "split", "suite", "source_config", "repo_id", "rows", "max_examples", "document_ids", "seed", "revision"),
          ("name", "split"), "dataset")
    if not all(isinstance(ds[key], str) and ds[key] for key in ("name", "split")):
        raise ValueError("dataset name and split must be strings")
    ds["name"] = ALIASES.get(ds["name"], ds["name"])
    if ("suite" in ds) == ("source_config" in ds):
        raise ValueError("dataset requires exactly one of suite or source_config")
    if "suite" in ds and set(ds) - {"name", "split", "suite"}:
        raise ValueError("a frozen suite cannot be resampled or have its source overridden")
    if "rows" in ds and "revision" in ds:
        raise ValueError("choose local rows or a Hub revision")
    if "document_ids" in ds:
        ids = ds["document_ids"]
        if not isinstance(ids, list) or not ids or not all(isinstance(mid, str) for mid in ids) or len(set(ids)) != len(ids):
            raise ValueError("document_ids must be a nonempty unique string array")
        if "max_examples" in ds:
            raise ValueError("choose document_ids or max_examples")
    for key in ("max_examples", "seed"):
        if key in ds:
            _integer(ds[key], f"dataset.{key}")
    if ds.get("max_examples", 0) < 0:
        raise ValueError("dataset.max_examples must be nonnegative")
    for key in ("suite", "source_config", "rows"):
        if key in ds:
            ds[key] = str((source.parent / ds[key]).resolve())
    if not isinstance(cfg["models"], list):
        raise ValueError("models must be an array of pinned model identities")
    for model in cfg["models"]:
        _keys(model, ("id", "version"), ("id", "version"), "model")
        if any(not isinstance(model[key], str) or not model[key] for key in ("id", "version")):
            raise ValueError("model id and version must be nonempty strings")
    ev = cfg.setdefault("evaluation", {})
    _keys(ev, ("bootstrap_samples", "confidence_level", "bootstrap_seed", "primary_metric", "historical_reports",
               "require_historical_parity"), (), "evaluation")
    for key, default in (("bootstrap_samples", 1000), ("confidence_level", .95), ("bootstrap_seed", 42),
                         ("historical_reports", []), ("require_historical_parity", False)):
        ev.setdefault(key, default)
    ev.setdefault("primary_metric", "primary")
    if ev["primary_metric"] not in {"primary", "binary", "binary_intra", "binary_inter", "entities", "intra", "inter"}:
        raise ValueError("unknown evaluation.primary_metric")
    validate_bootstrap(ev["bootstrap_samples"], ev["confidence_level"])
    _integer(ev["bootstrap_seed"], "evaluation.bootstrap_seed")
    if type(ev["require_historical_parity"]) is not bool:
        raise ValueError("require_historical_parity must be boolean")
    if not isinstance(ev["historical_reports"], list):
        raise ValueError("historical_reports must be an array")
    ev["historical_reports"] = [str((source.parent / path).resolve()) for path in ev["historical_reports"]]
    execution = cfg.setdefault("execution", {})
    _keys(execution, ("allow_network", "allow_paid", "error_policy"), (), "execution")
    for key in ("allow_network", "allow_paid"):
        execution.setdefault(key, False)
        if type(execution[key]) is not bool:
            raise ValueError(f"execution.{key} must be boolean")
    execution.setdefault("error_policy", "record")
    if execution["error_policy"] not in ("record", "raise"):
        raise ValueError("error_policy must be record or raise")
    cfg["output_dir"] = str((source.parent / cfg.get("output_dir", "logs/benchmark-runs")).resolve())
    plugins = cfg.setdefault("plugins", [])
    if not isinstance(plugins, list) or not all(isinstance(module, str) for module in plugins):
        raise ValueError("plugins must be an array of Python module names or .py paths")
    cfg["plugins"] = [str((source.parent / module).resolve()) if module.endswith(".py") else module
                      for module in plugins]

    def paths(value):
        if isinstance(value, dict):
            return {key: str((source.parent / Path(val).expanduser()).resolve()) if key in {"path", "cache_path"} and isinstance(val, str)
                    else paths(val) for key, val in value.items()}
        if isinstance(value, list):
            return [paths(val) for val in value]
        return value

    cfg["pipeline"] = paths(cfg["pipeline"])
    return ExperimentConfig(cfg, source)


def seed_configs(config):
    cfg = config.data
    for seed in cfg.get("seeds", [cfg.get("seed", 42)]):
        value = copy.deepcopy(cfg)
        value.pop("seeds", None)
        value["seed"] = seed
        yield ExperimentConfig(value, config.source)


def load_experiment(path):
    return resolve_config(read_yaml(path), path)


def load_suite(config):
    ds = config.data["dataset"]
    if "suite" in ds:
        suite = read_json(ds["suite"])
        documents(suite)
    else:
        cfg = read_json(ds["source_config"]) if ds["source_config"].endswith(".json") else read_yaml(ds["source_config"])
        if ds["name"] not in cfg["datasets"]:
            raise ValueError("dataset does not exist in source_config")
        cfg["active_dataset"] = ds["name"]
        cfg["datasets"][ds["name"]]["split"] = ds["split"]
        if "repo_id" in ds:
            cfg["datasets"][ds["name"]]["repo_id"] = ds["repo_id"]
        if "max_examples" in ds:
            cfg["datasets"][ds["name"]]["max_examples"] = ds["max_examples"]
        if "document_ids" in ds:
            cfg["datasets"][ds["name"]]["max_examples"] = 0
        if "seed" in ds:
            cfg["seed"] = ds["seed"]
        raw = read_jsonl(ds["rows"]) if "rows" in ds else None
        if "revision" in ds:
            from datasets import load_dataset
            raw = list(load_dataset(cfg["datasets"][ds["name"]]["repo_id"], split=ds["split"], revision=ds["revision"]))
        suite = freeze(cfg, raw)
        if "document_ids" in ds:
            from benchmark.io import fingerprint
            wanted = set(ds["document_ids"])
            suite["documents"] = [doc for doc in suite["documents"] if doc["id"] in wanted]
            if {doc["id"] for doc in suite["documents"]} != wanted:
                raise ValueError("document_ids includes IDs absent from the requested split")
            suite["selection"] = {"kind": "explicit-document-ids", "ids": ds["document_ids"]}
            suite["fingerprint"] = fingerprint({key: value for key, value in suite.items() if key != "fingerprint"})
    cfg = suite["config"]
    if cfg["active_dataset"] != ds["name"] or cfg["datasets"][ds["name"]]["split"] != ds["split"]:
        raise ValueError("experiment dataset/split differs from the frozen suite")
    return suite


def _merge(base, overrides):
    out = copy.deepcopy(base)
    for key, value in overrides.items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else copy.deepcopy(value)
    return out


def expand_campaign(path):
    payload, path = read_yaml(path), Path(path).resolve()
    _keys(payload, ("schema", "experiments"), ("schema", "experiments"), "campaign")
    if payload["schema"] != "workere-campaign/v1" or not isinstance(payload["experiments"], list) or not payload["experiments"]:
        raise ValueError("invalid campaign")
    for entry in payload["experiments"]:
        _keys(entry, ("config", "overrides"), ("config",), "campaign entry")
        source = (path.parent / entry["config"]).resolve()
        yield from seed_configs(resolve_config(_merge(read_yaml(source), entry.get("overrides", {})), source))


def expand_sweep(path):
    payload, path = read_yaml(path), Path(path).resolve()
    _keys(payload, ("schema", "base", "grid"), ("schema", "base", "grid"), "sweep")
    if payload["schema"] != "workere-sweep/v1" or not isinstance(payload["grid"], dict) or not payload["grid"]:
        raise ValueError("invalid sweep")
    source = (path.parent / payload["base"]).resolve()
    base = read_yaml(source)
    keys, values = list(payload["grid"]), list(payload["grid"].values())
    if any(not isinstance(v, list) or not v for v in values):
        raise ValueError("sweep grid values must be nonempty arrays")
    for index, combination in enumerate(itertools.product(*values)):
        value = copy.deepcopy(base)
        for key, variant in zip(keys, combination):
            node = value
            parts = key.split(".")
            for part in parts[:-1]:
                if not isinstance(node, dict) or part not in node:
                    raise ValueError(f"unknown sweep path: {key}")
                node = node[part]
            if not isinstance(node, dict) or parts[-1] not in node:
                raise ValueError(f"unknown sweep path: {key}")
            node[parts[-1]] = variant
        value["name"] += f"-sweep-{index:04d}"
        yield from seed_configs(resolve_config(value, source))
