"""Create a complete synthetic benchmark example, with no network or third-party packages."""
from __future__ import annotations

import argparse
from pathlib import Path

from benchmark.evaluate import compare, evaluate
from benchmark.examples.external import predict
from benchmark.io import write_json, write_jsonl
from benchmark.suite import freeze, model_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    out = Path(parser.parse_args().output_dir)
    cfg = {"active_dataset": "synthetic", "seed": 42, "datasets": {"synthetic": {
        "name": "SYNTHETIC demonstration", "repo_id": "local/synthetic", "split": "test",
        "text_field": "text", "ann_field": "annots", "max_examples": 0,
        "kfold": {"enabled": False, "n_folds": 1}, "constrain_to_pair_list": False}},
        "data": {"labels": ["causes", "causedby", "norel"], "binary_undirected": False,
                 "vote_to_binary": False, "binary_default": "norel", "novote_norel": True,
                 "shuffle_docs": False, "shuffle_pair_list": True},
        "experiment": {"resampling": {"enabled": False, "n_runs": 1, "tie_breaking": "norel"}}}
    rows = [{"id": "synthetic-rain", "tokens": ["Rain", "caused", "flooding", "."],
             "text": "<a Rain> caused <b flooding> .", "mentions": ["a", "b"],
             "spans": [[0], [2]], "sentences": [[0, 4]],
             "annots": "<a Rain> causes <b flooding>, <b flooding> causedby <a Rain>"}]
    suite = freeze(cfg, rows)
    inputs = list(model_inputs(suite))

    def lexical_rule(tokens, spans, source, target):
        a = " ".join(tokens[i] for i in spans[source]).lower()
        b = " ".join(tokens[i] for i in spans[target]).lower()
        return "causes" if (a, b) == ("rain", "flooding") else "causedby" if (b, a) == ("rain", "flooding") else "norel"

    write_json(out / "config.json", cfg)
    write_jsonl(out / "split.jsonl", rows)
    write_json(out / "suite.json", suite)
    write_jsonl(out / "inputs.jsonl", inputs)
    write_jsonl(out / "event-inputs.jsonl", model_inputs(suite, "event_mentions"))
    results = []
    for name, classifier in (("toy-norel", None), ("toy-lexical", lexical_rule)):
        preds = [predict(row, classifier) if classifier else predict(row) for row in inputs]
        result = evaluate(suite, preds, model=name)
        results.append(result)
        write_jsonl(out / f"{name}.predictions.jsonl", preds)
        write_json(out / f"{name}.result.json", result)
    write_json(out / "comparison.json", {"synthetic_only": True, "models": compare(results)})
    print(f"Synthetic example written to {out}")


if __name__ == "__main__":
    main()
