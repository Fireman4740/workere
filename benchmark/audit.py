"""Optional read-only schema audit of the exact repositories named in config.yaml.

python -m benchmark.audit --config config.yaml --output dataset-audit.json
Requires the existing datasets, huggingface_hub and PyYAML dependencies.
This never rebuilds datasets or replaces legacy repositories with corrected variants.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

from benchmark.io import write_json


def audit(repos):
    from datasets import load_dataset
    from huggingface_hub import HfApi

    api, results = HfApi(), []
    for name, repo in repos.items():
        record = {"dataset": name, "repo_id": repo}
        try:
            sha = api.dataset_info(repo).sha
            record["revision"] = sha
            ds = load_dataset(repo, revision=sha)
            splits = {}
            for split, dataset in ds.items():
                mentions = invalid = missing = 0
                for row in dataset:
                    tokens, spans, mids = row.get("tokens"), row.get("spans"), row.get("mentions")
                    if tokens is None or spans is None or mids is None or len(mids) != len(spans):
                        missing += 1
                        continue
                    mentions += len(mids)
                    invalid += sum(1 for span in spans if not span or
                                   any(type(i) is not int or not 0 <= i < len(tokens) for i in span))
                splits[split] = {"documents": len(dataset), "columns": dataset.column_names,
                                 "event_mentions": mentions, "invalid_spans": invalid,
                                 "documents_missing_span_annotations": missing,
                                 "typed_event_column": "mention_type" in dataset.column_names,
                                 "candidate_column": "pair_list" in dataset.column_names}
            record.update(status="available", splits=splits)
        except Exception as exc:
            record.update(status="unavailable", error=f"{type(exc).__name__}: {str(exc)[:300]}")
        results.append(record)
    return {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "datasets": results,
            "scope": "event mention spans only; no claim of generic or typed NER coverage"}


def main():
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    result = audit({name: ds["repo_id"] for name, ds in cfg["datasets"].items()})
    write_json(args.output, result)
    for row in result["datasets"]:
        print(row["dataset"], row["status"], {s: v["documents"] for s, v in row.get("splits", {}).items()})


if __name__ == "__main__":
    main()
