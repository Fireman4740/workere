"""Seeded document bootstrap and between-run variability, separate from metric definitions."""
from __future__ import annotations

import math
import random
import statistics

from benchmark.protocols import aggregate_statistics


def quantile(values, fraction):
    values = sorted(values)
    index = (len(values) - 1) * fraction
    low, high = math.floor(index), math.ceil(index)
    return values[low] + (values[high] - values[low]) * (index - low)


def validate_bootstrap(samples, confidence):
    if type(samples) is not int or samples < 0:
        raise ValueError("bootstrap_samples must be a nonnegative integer")
    if type(confidence) not in (float, int) or not math.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence_level must be in (0,1)")


def document_bootstrap(records, *, samples=1000, confidence=.95, seed=42):
    validate_bootstrap(samples, confidence)
    if not samples:
        return {"method": "disabled", "intervals": {}}
    if not records:
        raise ValueError("document bootstrap requires documents")
    rng, distributions = random.Random(seed), {}
    for _ in range(samples):
        # Sampling whole documents preserves correlations between their relation decisions.
        summary = aggregate_statistics([rng.choice(records) for _ in records])
        for group, values in summary.items():
            for metric in ("precision", "recall", "f1", "macro_f1"):
                if metric in values:
                    distributions.setdefault((group, metric), []).append(values[metric])
    alpha, intervals = (1 - confidence) / 2, {}
    for (group, metric), values in distributions.items():
        intervals.setdefault(group, {})[metric] = [quantile(values, alpha), quantile(values, 1 - alpha)]
    return {"method": "percentile-bootstrap-over-documents", "samples": samples,
            "confidence_level": confidence, "seed": seed, "documents": len(records), "intervals": intervals,
            "warnings": ["One document: degenerate interval; no generalization evidence."] if len(records) == 1 else []}


def run_variability(values, *, samples=1000, confidence=.95, seed=42):
    validate_bootstrap(samples, confidence)
    if not values:
        raise ValueError("run variability requires values")
    result = {"mean": statistics.mean(values), "runs": len(values),
              "sample_variance": statistics.variance(values) if len(values) > 1 else None,
              "sample_std": statistics.stdev(values) if len(values) > 1 else None,
              "mean_confidence_interval": None}
    if len(values) > 1 and samples:
        rng = random.Random(seed)
        means = [statistics.mean(rng.choices(values, k=len(values))) for _ in range(samples)]
        alpha = (1 - confidence) / 2
        result["mean_confidence_interval"] = [quantile(means, alpha), quantile(means, 1 - alpha)]
    return result
