"""Compare only matching suites/protocols, with separate document and repeated-seed intervals."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from benchmark.io import fingerprint, read_json
from benchmark.labels import BINARY_LABELS
from benchmark.protocols import historical_statistics, legacy_summary
from benchmark.statistics import document_bootstrap, run_variability


def load_result(path):
    path = Path(path)
    if path.is_dir():
        metadata, metrics = read_json(path / "metadata.json"), read_json(path / "metrics.json")
        if metadata["status"] not in {"completed", "completed_with_errors"}:
            raise ValueError("incomplete or failed runs cannot be compared")
        return {"path": str(path.resolve()), "name": metadata["name"], "seed": metadata["seed"],
                "recipe": metadata["recipe_fingerprint"], "status": metadata["status"],
                "resources": metadata.get("resources", {}), "historical_parity": metadata.get("historical_parity"),
                **metrics}
    payload = read_json(path)
    if payload.get("schema") != "workere-benchmark-result/v1" or payload.get("task") != "relations":
        raise ValueError("baseline must be imported with benchmark import-history onto the same suite")
    cfg, results = payload["config"], payload["results"]
    data = cfg["data"]
    labels = BINARY_LABELS if data["binary_undirected"] or data["vote_to_binary"] else data["labels"]
    stats = historical_statistics(results, labels)
    return {"path": str(path.resolve()), "name": payload["model"], "seed": None,
            "recipe": fingerprint({"baseline": payload["model"], "source": payload.get("historical_source")}),
            "protocol": "relations_gold", "protocol_id": payload["protocol"],
            "suite_fingerprint": payload["suite_fingerprint"], "status": "recorded_baseline",
            "historical_parity": payload.get("historical_source", {}).get("parity", "not_demonstrated"),
            "summary": legacy_summary(results), "document_statistics": stats,
            "uncertainty": document_bootstrap(stats), "resources": {"execution_mode": "historical",
                "api_cost_usd": None, "pipeline_seconds": None}}


def compare_runs(paths, *, metric="binary", samples=1000, confidence=.95, seed=42):
    if not paths:
        raise ValueError("provide at least one run")
    runs = [load_result(path) for path in paths]
    identity = [(r["suite_fingerprint"], r["protocol"], r["protocol_id"]) for r in runs]
    if any(key != identity[0] for key in identity):
        raise ValueError("incomparable datasets, splits, annotation snapshots or evaluation protocols")
    if any(metric not in run["summary"] for run in runs):
        raise ValueError(f"metric group {metric!r} unavailable for one of the runs")
    groups = defaultdict(list)
    for run in runs:
        groups[run["recipe"]].append(run)
    group_rows = []
    for recipe, group in groups.items():
        if len({run["seed"] for run in group if run["seed"] is not None}) != len([r for r in group if r["seed"] is not None]):
            raise ValueError("between-seed variance requires distinct seeds; duplicate seeds are decoding replicates")
        stats = {key: run_variability([r["summary"][metric][key] for r in group],
                                     samples=samples, confidence=confidence, seed=seed)
                 for key in ("precision", "recall", "f1")}
        group_rows.append({"recipe_fingerprint": recipe, "name": group[0]["name"],
                           "seeds": [r["seed"] for r in group], "metrics": stats})
    return {"schema": "workere-comparison/v2", "suite_fingerprint": identity[0][0],
            "protocol": identity[0][1], "metric": metric, "confidence_level": confidence,
            "variance_unit": "distinct pipeline seeds; sample variance", "groups": group_rows,
            "runs": [{key: run.get(key) for key in ("path", "name", "seed", "status", "summary",
                       "uncertainty", "resources", "historical_parity")} for run in runs]}


def markdown_table(comparison):
    metric = comparison["metric"]
    lines = [f"Protocol: `{comparison['protocol']}`; metric: `{metric}`; document bootstrap intervals.", "",
             "| Run | Seed | P | R | F1 | F1 CI | Intra F1 | Inter F1 | API cost USD | Pipeline s | Mode | Errors |",
             "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|---|"]
    def fmt(value):
        return "—" if value is None else f"{value:.4f}" if isinstance(value, (int, float)) else str(value)
    for row in comparison["runs"]:
        summary, resources = row["summary"], row["resources"]
        values = summary[metric]
        interval = row["uncertainty"].get("intervals", {}).get(metric, {}).get("f1")
        ci = "—" if interval is None else f"[{interval[0]:.4f}, {interval[1]:.4f}]"
        intra = summary.get("binary_intra" if metric == "binary" else "intra", {}).get("f1")
        inter = summary.get("binary_inter" if metric == "binary" else "inter", {}).get("f1")
        cells = [row["name"], row["seed"], values["precision"], values["recall"], values["f1"], ci,
                 intra, inter, resources.get("api_cost_usd"), resources.get("pipeline_seconds"),
                 resources.get("execution_mode"), "yes" if row["status"] == "completed_with_errors" else "no"]
        lines.append("| " + " | ".join(fmt(v).replace("|", "\\|") for v in cells) + " |")
    lines += ["", "Latency/cost describe this execution. Historical model costs are unknown unless recorded; "
              "replay wall time is not inference latency. Seed-mean intervals and variances are in the JSON."]
    return "\n".join(lines) + "\n"
