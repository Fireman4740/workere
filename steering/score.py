#!/usr/bin/env python
"""Score steering generations with the same binary, ordered-pair metric as the baseline."""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from eci.parse import ParseError, parse_triples
from eci.voting import aggregate
from steering.common import (
    aggregate_counts,
    binary_counts,
    binary_metric,
    format_metric,
    read_json,
    read_jsonl,
    triples_to_map,
    write_json,
)


def prediction_text(row: dict) -> str:
    for key in ("text", "output", "completion", "response"):
        value = row.get(key)
        if isinstance(value, str):
            return value
    raise ValueError(f"prediction row has no text field: {sorted(row)}")


def condition_of(row: dict) -> str:
    return str(row.get("condition") or row.get("direction") or f"alpha={row.get('alpha', 0)}")


def reference_metric(reference: dict, gold: dict, baseline: str, ids) -> dict | None:
    """Baseline metric on a subset the reference file has no precomputed entry for."""
    stored = reference.get(baseline, {})
    counts = []
    for doc_id in ids:
        if doc_id not in stored:
            continue
        pred = {tuple(k.split(",", 1)): label for k, label in stored[doc_id].items()}
        counts.append(binary_counts(pred, gold[doc_id]))
    return binary_metric(aggregate_counts(counts)) if counts else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--include-truncated", action="store_true")
    args = parser.parse_args()

    reference = read_json(args.reference)
    gold = {
        doc_id: {(key.split(",", 1)[0], key.split(",", 1)[1]): label for key, label in mapping.items()}
        for doc_id, mapping in reference["gold"].items()
    }
    subsets = {
        "n13_50": reference["ids"],
        "qwen_47": reference["qwen_ids"],
        "haiku_50": reference["haiku_ids"],
        "paired_42": reference["paired"],
        "exploratory_3": [d for d in reference["ids"] if d not in set(reference["qwen_ids"])],
    }

    grouped = defaultdict(lambda: defaultdict(list))
    excluded = defaultdict(lambda: defaultdict(int))
    for row in read_jsonl(args.predictions):
        condition = condition_of(row)
        doc_id = row["doc_id"]
        if row.get("finish_reason") not in (None, "stop") and not args.include_truncated:
            excluded[condition][doc_id] += 1
            continue
        grouped[condition][doc_id].append(row)

    run_docs = sorted({doc for per_doc in grouped.values() for doc in per_doc} & set(reference["ids"]))
    subsets = {"run_docs": run_docs, **subsets}

    report = {"conditions": {}, "excluded": {}}
    csv_rows = []
    for condition, per_doc in sorted(grouped.items()):
        predictions = {}
        parse_errors = defaultdict(int)
        pass_counts = defaultdict(int)
        for doc_id, rows in per_doc.items():
            passes = []
            for row in sorted(rows, key=lambda item: int(item.get("pass", 0))):
                pass_counts[doc_id] += 1
                try:
                    passes.append(parse_triples(prediction_text(row)))
                except (ParseError, ValueError):
                    parse_errors[doc_id] += 1
            if not passes:
                continue
            final, _ = aggregate(passes, "norel", random.Random(f"steer::{doc_id}"))
            predictions[doc_id] = triples_to_map(final)

        condition_report = {}
        for subset_name, ids in subsets.items():
            counts = []
            per_doc_rows = []
            for doc_id in ids:
                if doc_id not in predictions:
                    continue
                row = binary_counts(predictions[doc_id], gold[doc_id])
                counts.append(row)
                per_doc_rows.append({"doc": doc_id, **row})
            metric = binary_metric(aggregate_counts(counts))
            metric["documents"] = len(per_doc_rows)
            metric["missing"] = len(ids) - len(per_doc_rows)
            metric["parse_errors"] = sum(parse_errors.get(doc_id, 0) for doc_id in ids)
            metric["passes"] = sum(pass_counts.get(doc_id, 0) for doc_id in ids)
            metric["per_document"] = per_doc_rows
            for baseline in ("qwen", "haiku"):
                key = f"{subset_name}/{baseline}"
                baseline_metric = reference["metrics"].get(key) or reference_metric(
                    reference, gold, baseline, ids
                )
                metric[f"baseline_{baseline}_f1"] = baseline_metric["f1"] if baseline_metric else None
                metric[f"delta_{baseline}_f1"] = (
                    metric["f1"] - baseline_metric["f1"] if baseline_metric else None
                )
            condition_report[subset_name] = metric
            for row in per_doc_rows:
                csv_rows.append({
                    "condition": condition,
                    "subset": subset_name,
                    "doc": row["doc"],
                    **{key: row[key] for key in ("tp", "fp", "fn", "support_pos", "predicted_pos")},
                    "f1": 2 * row["tp"] / (2 * row["tp"] + row["fp"] + row["fn"])
                    if 2 * row["tp"] + row["fp"] + row["fn"] else 0.0,
                })
        report["conditions"][condition] = condition_report
        report["excluded"][condition] = dict(excluded.get(condition, {}))
        print(f"\n{condition}")
        for subset_name, metric in condition_report.items():
            print(
                f"  {subset_name}: docs={metric['documents']} missing={metric['missing']} "
                f"{format_metric(metric)} "
                f"delta_qwen={100 * (metric['delta_qwen_f1'] or 0):+.2f} "
                f"delta_haiku={100 * (metric['delta_haiku_f1'] or 0):+.2f}"
            )

    write_json(args.out, report)
    path = Path(args.csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]) if csv_rows else [
            "condition", "subset", "doc", "tp", "fp", "fn", "support_pos",
            "predicted_pos", "f1",
        ])
        writer.writeheader()
        writer.writerows(csv_rows)


if __name__ == "__main__":
    main()
