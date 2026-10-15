"""Read-only replay of persisted checkpoints and report rows. Never calls a provider."""
from __future__ import annotations

import math
import hashlib
from collections import defaultdict
from pathlib import Path

from benchmark import report as R
from benchmark.io import read_json, read_jsonl
from benchmark.labels import BINARY_LABELS

HEADLINES = ("micro_precision", "micro_recall", "micro_f1", "macro_f1", "total_pairs",
             "binary", "binary_intra", "binary_inter")
METRICS = HEADLINES + ("per_label", "sentence_unknown_pairs", "sentence_unknown_pct",
                      "per_doc_metrics", "per_lang_metrics")


def score_rows(rows, labels):
    """Same arithmetic and grouping as report.build, using the persisted scoring universe."""
    def scope(group):
        return R._scope_metrics([r["gold"] for r in group], [r["pred"] for r in group],
                                [r.get("sentence_relation", "unknown") for r in group], labels)

    groups, langs = defaultdict(list), defaultdict(list)
    for row in rows:
        groups[row["id"]].append(row)
        langs[row.get("lang", "eng")].append(row)
    g = scope(rows)
    out = dict(g["multiclass"])
    out.pop("total")
    out.update({k: g[k] for k in ("binary", "binary_intra", "binary_inter", "sentence_unknown_pairs")})
    out["total_pairs"] = len(rows)
    out["sentence_unknown_pct"] = g["sentence_unknown_pairs"] / len(rows) if rows else 0.0
    per_doc = []
    for doc_id, group in groups.items():
        d = scope(group)
        per_doc.append({"id": doc_id, "doc_idx": group[0].get("doc_idx"), "pairs": len(group),
                        **d["multiclass"], **{k: d[k] for k in
                        ("binary", "binary_intra", "binary_inter", "sentence_unknown_pairs")}})
    out["per_doc_metrics"] = per_doc
    out["per_lang_metrics"] = {lang: scope(group) for lang, group in langs.items()}
    return out


def _differences(expected, actual, path="", *, atol=1e-12):
    differences = []
    checks = 0
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return 0, [{"path": path, "reason": "missing object"}]
        for key, value in expected.items():
            if key not in actual:
                differences.append({"path": f"{path}.{key}", "reason": "missing field"})
            else:
                n, delta = _differences(value, actual[key], f"{path}.{key}", atol=atol)
                checks += n
                differences.extend(delta)
    elif isinstance(expected, list):
        if not isinstance(actual, list):
            return 0, [{"path": path, "reason": "missing array"}]
        if all(isinstance(r, dict) and "id" in r for r in expected):
            wanted = {r["id"]: r for r in expected}
            got = {r["id"]: r for r in actual}
            if len(wanted) != len(expected) or len(got) != len(actual):
                differences.append({"path": path, "reason": "duplicate document IDs"})
            if set(wanted) != set(got):
                differences.append({"path": path, "reason": "document coverage mismatch"})
            for key, value in wanted.items():
                n, delta = _differences(value, got.get(key), f"{path}[{key}]", atol=atol)
                checks += n
                differences.extend(delta)
        else:
            checks += 1
            if expected != actual:
                differences.append({"path": path, "expected": expected, "actual": actual})
    else:
        checks += 1
        same = (isinstance(actual, (int, float)) and math.isfinite(expected)
                and math.isfinite(actual) and math.isclose(expected, actual, rel_tol=0, abs_tol=atol)
                if isinstance(expected, (int, float)) else expected == actual)
        if not same:
            differences.append({"path": path, "expected": expected, "actual": actual})
    return checks, differences


def verify(report_path, checkpoint=None):
    path = Path(report_path)
    payload = read_json(path)
    cfg, logged = payload["config"], payload["results"]
    missing = [key for key in HEADLINES if key not in logged]
    if missing:
        raise ValueError(f"{path}: missing historical headline metrics {missing}")
    data = cfg["data"]
    labels = (BINARY_LABELS if data["binary_undirected"] or data.get("vote_to_binary", False)
              else data["labels"])
    if checkpoint is None and (path.parent / "docs.jsonl").is_file():
        checkpoint = path.parent / "docs.jsonl"
    rows = logged["per_pair_predictions"]
    if checkpoint:
        records = read_jsonl(checkpoint)
        if len({r["id"] for r in records}) != len(records):
            raise ValueError("duplicate document IDs in checkpoint")
        rebuilt = R.build(records, len(records) + logged.get("skipped_docs", 0), cfg, labels)
        level = "checkpoint-to-report"
    else:
        rebuilt = score_rows(rows, labels)
        # Rows cannot identify empty documents; retain their IDs, then recompute zero metrics.
        present = {r["id"] for r in rebuilt["per_doc_metrics"]}
        for doc in logged.get("per_doc_metrics", []):
            if doc["id"] not in present:
                empty = R._scope_metrics([], [], [], labels)
                rebuilt["per_doc_metrics"].append({"id": doc["id"], "doc_idx": doc.get("doc_idx"),
                    "pairs": 0, **empty["multiclass"], **{k: empty[k] for k in
                    ("binary", "binary_intra", "binary_inter", "sentence_unknown_pairs")}})
        level = "report-rows-only"
    wanted = {key: logged[key] for key in METRICS if key in logged}
    checks, differences = _differences(wanted, rebuilt)
    if checkpoint:
        # Also verify the actual scored pairs, labels, vote counts, scope and distances.
        expected_rows = sorted(rows, key=lambda r: (r["id"], r["pair"]))
        actual_rows = sorted(rebuilt["per_pair_predictions"], key=lambda r: (r["id"], r["pair"]))
        if len(expected_rows) != len(actual_rows):
            differences.append({"path": "per_pair_predictions", "reason": "row count mismatch"})
        for i, (old, new) in enumerate(zip(expected_rows, actual_rows)):
            n, delta = _differences(old, new, f"per_pair_predictions[{i}]")
            checks += n
            differences.extend(delta)
    return {"report": str(path.resolve()), "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "checkpoint": str(Path(checkpoint).resolve()) if checkpoint else None,
            "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() if checkpoint else None,
            "level": level, "checks": checks, "status": "different" if differences else "equal",
            "differences": differences, "binary_f1": rebuilt["binary"]["f1"]}


def verify_many(paths):
    if not paths:
        raise ValueError("no historical reports supplied; parity is NOT demonstrated")
    results = [verify(path) for path in paths]
    return {"schema": "workere-benchmark-parity/v1", "absolute_tolerance": 1e-12,
            "status": "equal" if all(r["status"] == "equal" for r in results) else "different",
            "reports": results}


def import_history(suite, path, *, model):
    """Import persisted final decisions onto a frozen suite, only if ALL metrics agree.

    Explicit negative rows preserve the original scored pair universe, including pairs
    rejected by voting. This imports final decisions, not the unrecorded inference passes.
    """
    from benchmark.evaluate import evaluate
    from benchmark.suite import documents, evaluation_config

    payload = read_json(path)
    if evaluation_config(payload["config"]) != suite["config"]:
        raise ValueError("historical evaluation configuration differs from the frozen suite")
    logged = payload["results"]
    missing = [key for key in HEADLINES if key not in logged]
    if missing:
        raise ValueError(f"missing historical headline metrics: {missing}")
    by_id = {d.id: {"id": d.id, "triples": []} for d in documents(suite)}
    if set(by_id) != {d["id"] for d in logged["per_doc_metrics"]}:
        raise ValueError("historical document coverage differs from the frozen suite")
    for row in logged["per_pair_predictions"]:
        source, target = row["pair"].split(",", 1)
        by_id[row["id"]]["triples"].append([source, row["pred"], target])
    result = evaluate(suite, by_id.values(), model=model, representation="final")
    wanted = {key: logged[key] for key in METRICS if key in logged}
    checks, delta = _differences(wanted, result["results"])
    # Compare gold and sentence scopes too: equal aggregate scores can conceal dataset drift.
    old = {(r["id"], r["pair"]): (r["gold"], r.get("sentence_relation", "unknown"))
           for r in logged["per_pair_predictions"]}
    new = {(r["id"], r["pair"]): (r["gold"], r["sentence_relation"])
           for r in result["results"]["per_pair_predictions"]}
    if delta or old != new:
        raise ValueError(f"historical parity failed: {len(delta)} metric discrepancies; "
                         f"gold/scope agreement={old == new}")
    result["historical_source"] = {"report": str(Path(path).resolve()),
                                    "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(), "checks": checks,
                                    "parity": "equal"}
    return result
