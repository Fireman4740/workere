"""Shared data and scoring helpers for the N13 CTB50 steering pilot."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

from eci.labels import is_norel


def read_json(path: str | Path):
    return json.loads(Path(path).read_text())


def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n")


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def load_suite(path: str | Path) -> dict[str, dict]:
    suite = read_json(path)
    documents = suite["documents"] if isinstance(suite, dict) else suite
    return {document["id"]: document for document in documents}


def gold_map(document: dict) -> dict[tuple[str, str], str]:
    return {(source, target): label for source, label, target in document["gold"]}


def triples_to_map(triples: Iterable[Iterable[str]]) -> dict[tuple[str, str], str]:
    return {(source, target): label for source, label, target in triples}


def binary_counts(
    pred: Mapping[tuple[str, str], str],
    gold: Mapping[tuple[str, str], str],
) -> dict[str, int]:
    gold_positive = {pair for pair, label in gold.items() if not is_norel(label)}
    pred_positive = {pair for pair, label in pred.items() if not is_norel(label)}
    return {
        "tp": len(gold_positive & pred_positive),
        "fp": len(pred_positive - gold_positive),
        "fn": len(gold_positive - pred_positive),
        "support_pos": len(gold_positive),
        "predicted_pos": len(pred_positive),
    }


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def binary_metric(counts: Mapping[str, int]) -> dict[str, float]:
    precision, recall, f1 = prf(counts["tp"], counts["fp"], counts["fn"])
    return {
        **counts,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def aggregate_counts(rows: Iterable[Mapping[str, int]]) -> dict[str, int]:
    total = {"tp": 0, "fp": 0, "fn": 0, "support_pos": 0, "predicted_pos": 0}
    for row in rows:
        for key in total:
            total[key] += int(row.get(key, 0))
    return total


def metric_row(counts: Mapping[str, int], **extra) -> dict:
    return {**binary_metric(counts), **extra}


def format_metric(metric: Mapping[str, float]) -> str:
    return (
        f"TP {metric['tp']} FP {metric['fp']} FN {metric['fn']} "
        f"P {100 * metric['precision']:.2f} R {100 * metric['recall']:.2f} "
        f"F1 {100 * metric['f1']:.2f}"
    )
