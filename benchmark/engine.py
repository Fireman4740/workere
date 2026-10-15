"""Execute a resolved pipeline, persist its public contract, then evaluate offline."""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import inspect
import json
import platform
import random
import subprocess
import sys
import time
import traceback
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from benchmark.configuration import load_suite
from benchmark.history import import_history, verify_many
from benchmark.interfaces import Context
from benchmark.io import fingerprint, read_json, read_jsonl, write_json
from benchmark.pipeline import build_pipeline
from benchmark.protocols import public_inputs, score_predictions
from benchmark.registry import load_plugins
from benchmark.schema import Prediction
from benchmark.statistics import document_bootstrap, quantile


class ExecutionError(RuntimeError):
    def __init__(self, run_dir, reason):
        self.run_dir = Path(run_dir)
        super().__init__(f"{reason} (run: {run_dir})")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _append(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()


def _replace_json(path, value):
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    write_json(temp, value)
    temp.replace(path)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assets(value, prefix="pipeline"):
    out = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "path" and isinstance(child, str):
                path = Path(child)
                out.append({"parameter": f"{prefix}.{key}", "path": child,
                            "sha256": _sha(path) if path.is_file() else None,
                            "bytes": path.stat().st_size if path.is_file() else None})
            else:
                out += _assets(child, f"{prefix}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            out += _assets(child, f"{prefix}.{index}")
    return out


def _source_receipt(run_dir, manifest):
    package = Path(__file__).resolve().parent
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=package, text=True,
                                         stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=package, text=True))
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
        revision = package / "COMMIT.txt"
        if revision.is_file():
            value = revision.read_text().strip()
            if len(value) == 40 and all(c in "0123456789abcdef" for c in value):
                commit, dirty = value, False
    files = {f"benchmark/{path.relative_to(package)}": path for path in package.rglob("*.py")
             if not {".venv", "build", "__pycache__"} & set(path.relative_to(package).parts)}
    for index, item in enumerate(manifest):
        module_name = item["implementation"].split(":", 1)[0]
        module = sys.modules.get(module_name)
        path = getattr(module, "__file__", None)
        if path and Path(path).is_file() and not Path(path).is_relative_to(package):
            source = Path(path)
            if (source.parent / "__init__.py").is_file():
                for child in source.parent.rglob("*.py"):
                    files[f"external_plugins/{index}/{source.parent.name}/{child.relative_to(source.parent)}"] = child
            else:
                files[f"external_plugins/{index}_{source.name}"] = source
    for parent in (package, package.parent):
        for name in ("pyproject.toml", "uv.lock"):
            path = parent / name
            if path.is_file():
                files[f"manifests/{parent.name}/{name}"] = path
    checksums = {}
    with zipfile.ZipFile(run_dir / "source.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in sorted(files.items()):
            archive.write(path, name)
            checksums[name] = _sha(path)
    return {"git_commit": commit, "git_dirty": dirty, "python": sys.version,
            "platform": platform.platform(), "sources_sha256": checksums,
            "installed_packages": dict(sorted((dist.metadata["Name"], dist.version)
                                               for dist in importlib.metadata.distributions()
                                               if dist.metadata["Name"]))}


def _history(config, suite):
    ev = config["evaluation"]
    paths = list(ev["historical_reports"])
    monolithic = config["pipeline"].get("monolithic", {})
    legacy = monolithic.get("plugin") == "legacy_report"
    if legacy:
        if config["protocol"] != "relations_gold":
            raise ValueError("legacy_report requires the relations_gold protocol")
        source = monolithic.get("params", {}).get("path")
        if source not in paths:
            paths.append(source)
    if not paths:
        if ev["require_historical_parity"]:
            raise ValueError("historical parity required but no original reports were supplied")
        return {"status": "not_demonstrated", "reports": []}
    receipt = verify_many(paths)
    if receipt["status"] != "equal":
        raise ValueError("historical score parity failed")
    # Historic baselines may only be compared to the same configuration, gold and scopes.
    if config["protocol"] == "relations_gold":
        for path in paths:
            import_history(suite, path, model="parity-check")
    receipt["scope"] = "reference-mention relations only; no historical end-to-end baseline"
    return receipt


def execute(config, *, output_root=None):
    """One run, one effective seed. Failed documents are recorded and scored as empty."""
    import yaml

    cfg = copy.deepcopy(config.data)
    if "seeds" in cfg:
        raise ValueError("expand seed_configs before execute")
    root = Path(output_root or cfg["output_dir"])
    run_id = f"{cfg['name']}-seed-{cfg['seed']}-{uuid.uuid4().hex[:12]}"
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    metadata = {"schema": "workere-run/v2", "run_id": run_id, "name": cfg["name"], "status": "running",
                "started_utc": _now(), "config_source": str(config.source), "seed": cfg["seed"],
                "protocol": cfg["protocol"], "models": cfg["models"], "errors": 0}
    write_json(run_dir / "metadata.json", metadata)
    write_json(run_dir / "parameters.json", cfg)
    if config.source.is_file():
        with (run_dir / "source-config.yaml").open("xb") as stream:
            stream.write(config.source.read_bytes())
    with (run_dir / "parameters.yaml").open("x", encoding="utf-8") as stream:
        yaml.safe_dump(cfg, stream, allow_unicode=True, sort_keys=False)
    pipeline, resources = None, []
    try:
        suite = load_suite(config)
        metadata["suite_fingerprint"] = suite["fingerprint"]
        write_json(run_dir / "suite.json", suite)
        receipt = _history(cfg, suite)
        write_json(run_dir / "historical-parity.json", receipt)
        metadata["historical_parity"] = receipt["status"]
        registry = load_plugins(cfg["plugins"])
        random.seed(cfg["seed"])
        setup = time.perf_counter()
        pipeline, manifest = build_pipeline(cfg["pipeline"], cfg["protocol"], registry, cfg["execution"])
        metadata["setup_seconds"] = time.perf_counter() - setup
        metadata["components"] = manifest
        assets = _assets(cfg["pipeline"])
        metadata["assets"] = assets
        metadata["recipe_fingerprint"] = fingerprint({"suite": suite["fingerprint"], "protocol": cfg["protocol"],
            "pipeline": cfg["pipeline"], "plugins": cfg["plugins"], "models": cfg["models"],
            "assets": assets})
        write_json(run_dir / "environment.json", _source_receipt(run_dir, manifest))
        inputs = list(public_inputs(suite, cfg["protocol"]))
        all_rows = []
        with (run_dir / "predictions.jsonl").open("x", encoding="utf-8") as predictions, \
                (run_dir / "inputs.jsonl").open("x", encoding="utf-8") as input_stream, \
                (run_dir / "errors.jsonl").open("x", encoding="utf-8") as errors, \
                (run_dir / "decisions.jsonl").open("x", encoding="utf-8") as decisions, \
                (run_dir / "usage.jsonl").open("x", encoding="utf-8") as usage:
            served_models, providers = set(), set()
            for item in inputs:
                doc = item.document
                context = Context(cfg["seed"], cfg["protocol"], doc.id)
                context.usage_sink = lambda event: _append(decisions, {"document_id": doc.id, **event})
                _append(input_stream, item.to_dict())
                before, error = time.perf_counter(), None
                try:
                    prediction = pipeline.predict(item, context).validate(doc)
                    if cfg["protocol"] == "relations_gold" and (
                            {e.id: e.token_indices for e in prediction.entities} !=
                            {e.id: e.token_indices for e in item.reference_entities}):
                        raise ValueError("relations_gold pipeline changed reference mentions")
                    row = prediction.to_dict(doc)
                except Exception as exc:
                    error = {"id": doc.id, "type": type(exc).__name__, "message": str(exc)[:4000],
                             "traceback": traceback.format_exc()}
                    _append(errors, error)
                    metadata["errors"] += 1
                    if cfg["execution"]["error_policy"] == "raise":
                        raise
                    # Preserve coverage: failures are false negatives, not silently skipped docs.
                    row = Prediction(doc.id, item.reference_entities or (), metadata={"error": error["message"]}).to_dict(doc)
                elapsed = time.perf_counter() - before
                cost = context.cost_usd
                if any(plugin["requires_network"] for plugin in manifest) and not context.usage:
                    cost = None
                calls = sum(bool(event.get("api_call", True)) and not event.get("cache_hit", False)
                            for event in context.usage)
                for event in context.usage:
                    response = event.get("response") or {}
                    if response.get("model"):
                        served_models.add(response["model"])
                    if response.get("provider"):
                        providers.add(response["provider"])
                resource = {"id": doc.id, "latency_seconds": elapsed, "api_cost_usd": cost,
                            "status": "error" if error else "ok", "calls": calls,
                            "decision_events": len(context.usage),
                            "cache_hits": sum(event.get("cache_hit", False) for event in context.usage)}
                resources.append(resource)
                _append(usage, {**resource, "provider_usage": context.usage})
                _append(predictions, row)
                all_rows.append(row)
        result = score_predictions(suite, all_rows, cfg["protocol"], model=cfg["name"])
        ev = cfg["evaluation"]
        if ev["primary_metric"] not in result["summary"]:
            raise ValueError("selected primary metric unavailable for this protocol")
        metadata["primary_metric"] = ev["primary_metric"]
        result["uncertainty"] = document_bootstrap(result["document_statistics"], samples=ev["bootstrap_samples"],
                                                   confidence=ev["confidence_level"], seed=ev["bootstrap_seed"])
        write_json(run_dir / "metrics.json", result)
        latencies = [row["latency_seconds"] for row in resources]
        principal = next((m for m in manifest if m["kind"] == "pipeline"), None)
        principal = principal or next((m for m in manifest if m["kind"] == "relation_predictor"), manifest[0])
        metadata["resources"] = {"execution_mode": principal["execution_mode"],
            "api_calls": sum(row["calls"] for row in resources),
            "cache_hits": sum(row["cache_hits"] for row in resources),
            "decision_events": sum(row["decision_events"] for row in resources),
            "api_cost_usd": sum(row["api_cost_usd"] for row in resources) if all(row["api_cost_usd"] is not None for row in resources) else None,
            "known_api_cost_usd": sum(row["api_cost_usd"] or 0 for row in resources),
            "cost_coverage_documents": sum(row["api_cost_usd"] is not None for row in resources),
            "documents": len(resources), "pipeline_seconds": sum(latencies),
            "mean_document_seconds": sum(latencies) / len(latencies),
            "p50_document_seconds": quantile(latencies, .5), "p95_document_seconds": quantile(latencies, .95)}
        metadata["served_models"], metadata["providers"] = sorted(served_models), sorted(providers)
        metadata["predictions_sha256"] = _sha(run_dir / "predictions.jsonl")
        metadata["metrics_sha256"] = _sha(run_dir / "metrics.json")
        metadata["status"] = "completed_with_errors" if metadata["errors"] else "completed"
    except BaseException as exc:
        metadata["status"] = "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
        metadata["error"] = {"type": type(exc).__name__, "message": str(exc)[:4000]}
        write_json(run_dir / "failure.json", {**metadata["error"], "traceback": traceback.format_exc()})
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        raise ExecutionError(run_dir, str(exc)) from exc
    finally:
        if pipeline is not None:
            try:
                pipeline.close()
            except Exception as exc:
                metadata["cleanup_error"] = {"type": type(exc).__name__, "message": str(exc)[:1000]}
        journal = run_dir / "decisions.jsonl"
        if "resources" not in metadata and journal.exists():
            # Fail-fast must retain accounting for successful and failed attempts already made.
            events = read_jsonl(journal)
            costs = [e.get("cost_usd") for e in events]
            metadata["resources"] = {"api_calls": sum(bool(e.get("api_call", True)) for e in events),
                "cache_hits": sum(bool(e.get("cache_hit")) for e in events), "decision_events": len(events),
                "known_api_cost_usd": sum(c or 0 for c in costs),
                "api_cost_usd": sum(costs) if all(c is not None for c in costs) else None,
                "accounting_scope": "persisted decision attempts; evaluation incomplete"}
        metadata["finished_utc"] = _now()
        metadata["total_seconds"] = time.perf_counter() - started
        _replace_json(run_dir / "metadata.json", metadata)
    return {"run_dir": str(run_dir.resolve()), "status": metadata["status"],
            "primary_f1": result["summary"][cfg["evaluation"]["primary_metric"]]["f1"], "errors": metadata["errors"]}


def reevaluate(run_dir, output, *, samples=None, confidence=None, seed=None):
    """Re-evaluate persisted records. Never load plugins or instantiate model components."""
    run_dir = Path(run_dir)
    metadata = read_json(run_dir / "metadata.json")
    if metadata.get("predictions_sha256") and metadata["predictions_sha256"] != _sha(run_dir / "predictions.jsonl"):
        raise ValueError("saved predictions checksum mismatch")
    cfg = read_json(run_dir / "parameters.json")
    result = score_predictions(read_json(run_dir / "suite.json"), read_jsonl(run_dir / "predictions.jsonl"),
                               cfg["protocol"], model=cfg["name"])
    ev = cfg["evaluation"]
    result["uncertainty"] = document_bootstrap(result["document_statistics"],
        samples=ev["bootstrap_samples"] if samples is None else samples,
        confidence=ev["confidence_level"] if confidence is None else confidence,
        seed=ev["bootstrap_seed"] if seed is None else seed)
    write_json(output, result)
    return result
