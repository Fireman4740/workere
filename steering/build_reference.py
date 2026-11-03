#!/usr/bin/env python
"""Build the N13 CTB50 reference: gold, Qwen 7-pass baseline, Haiku 5.5 baseline.

The N13 set has 50 documents. Qwen completed 47 of them in the b41468ee run; the
other three are the known truncation failures. Haiku completed all 50. This script
keeps the per-document ordered-pair maps so later steering conditions can be
scored against exactly the same gold and baselines.
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

from steering.common import (
    aggregate_counts,
    binary_metric,
    binary_counts,
    format_metric,
    gold_map,
    load_suite,
    read_json,
    read_jsonl,
    triples_to_map,
    write_json,
)


def haiku_predictions(runs_glob: str) -> dict[str, dict]:
    out = {}
    for directory in glob.glob(runs_glob):
        base = os.path.basename(directory)
        prefix = "ctb-h1-haiku-"
        marker = "-seed-42-"
        if not base.startswith(prefix) or marker not in base:
            continue
        doc_id = base[len(prefix):].split(marker)[0]
        rows = read_jsonl(Path(directory) / "predictions.jsonl")
        if len(rows) != 1 or rows[0].get("id") != doc_id:
            raise ValueError(f"unexpected Haiku prediction file: {directory}")
        out[doc_id] = {
            (row["source"], row["target"]): row["label"]
            for row in rows[0]["relations"]
        }
    return out


def qwen_predictions(path: str) -> dict[str, dict]:
    out = {}
    for row in read_jsonl(path):
        if row.get("failure") is not None:
            continue
        out[row["id"]] = triples_to_map(row["pred_triples"])
    return out


def score_docs(ids, gold, predictions):
    rows = []
    for doc_id in ids:
        if doc_id not in predictions:
            continue
        counts = binary_counts(predictions[doc_id], gold[doc_id])
        rows.append({"doc": doc_id, **counts})
    return aggregate_counts(rows), rows


def map_to_json(mapping: dict[tuple[str, str], str]) -> dict[str, str]:
    return {f"{source},{target}": label for (source, target), label in mapping.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--ids", required=True)
    parser.add_argument("--qwen-docs", required=True)
    parser.add_argument("--haiku-runs", required=True)
    parser.add_argument("--paired", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    suite = load_suite(args.suite)
    ids = [row["id"] for row in read_json(args.ids)]
    paired = set(read_json(args.paired)["documents"])
    gold = {doc_id: gold_map(suite[doc_id]) for doc_id in ids}
    qwen = qwen_predictions(args.qwen_docs)
    haiku = haiku_predictions(args.haiku_runs)

    qwen_ids = [doc_id for doc_id in ids if doc_id in qwen]
    haiku_ids = [doc_id for doc_id in ids if doc_id in haiku]
    paired_ids = [doc_id for doc_id in ids if doc_id in paired]
    subsets = {
        "n13_50": ids,
        "qwen_47": qwen_ids,
        "haiku_50": haiku_ids,
        "paired_42": paired_ids,
    }
    metrics = {}
    for name, subset in subsets.items():
        for model, predictions in (("qwen", qwen), ("haiku", haiku)):
            counts, per_doc = score_docs(subset, gold, predictions)
            metrics[f"{name}/{model}"] = {
                **binary_metric(counts),
                "documents": len(per_doc),
                "missing": len(subset) - len(per_doc),
                "per_document": per_doc,
            }
    reference = {
        "ids": ids,
        "paired": sorted(paired_ids),
        "qwen_ids": qwen_ids,
        "haiku_ids": haiku_ids,
        "gold": {doc_id: map_to_json(gold[doc_id]) for doc_id in ids},
        "qwen": {doc_id: map_to_json(qwen[doc_id]) for doc_id in qwen_ids},
        "haiku": {doc_id: map_to_json(haiku[doc_id]) for doc_id in haiku_ids},
        "metrics": metrics,
    }
    write_json(args.out, reference)
    for key, metric in metrics.items():
        print(f"{key}: docs={metric['documents']} missing={metric['missing']} {format_metric(metric)}")


if __name__ == "__main__":
    main()
