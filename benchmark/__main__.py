"""CLI: python -m benchmark --help. Scoring and replay use only the standard library."""
from __future__ import annotations

import argparse
import json

from benchmark.evaluate import compare, evaluate, evaluate_event_mentions
from benchmark.history import import_history, verify_many
from benchmark.io import read_json, read_jsonl, write_json, write_jsonl
from benchmark.suite import freeze, model_inputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("freeze", help="freeze the configured split and candidate universe")
    p.add_argument("--config", required=True)
    p.add_argument("--rows", help="optional JSONL of the entire split, in original order")
    p.add_argument("--output", required=True)
    p = commands.add_parser("inputs", help="export model inputs without gold relations")
    p.add_argument("--suite", required=True)
    p.add_argument("--task", choices=("relations", "event_mentions"), default="relations")
    p.add_argument("--output", required=True)
    p = commands.add_parser("score", help="score document-level JSONL predictions")
    p.add_argument("--suite", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--task", choices=("relations", "event_mentions"), default="relations")
    p.add_argument("--representation", choices=("final", "passes"), default="final")
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("verify-history", help="fail unless persisted scores are reproduced")
    p.add_argument("reports", nargs="+")
    p.add_argument("--output", required=True)
    p = commands.add_parser("compare", help="compare only matching suites and protocols")
    p.add_argument("reports", nargs="+")
    p.add_argument("--output", required=True)
    p = commands.add_parser("import-history", help="import final decisions only after suite parity")
    p.add_argument("--suite", required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "freeze":
            if args.config.endswith(".json"):
                cfg = read_json(args.config)
            else:
                import yaml  # Only config YAML import needs PyYAML, never prompt validation.
                with open(args.config, encoding="utf-8") as stream:
                    cfg = yaml.safe_load(stream)
            result = freeze(cfg, read_jsonl(args.rows) if args.rows else None)
        elif args.command == "inputs":
            write_jsonl(args.output, model_inputs(read_json(args.suite), args.task))
            return 0
        elif args.command == "score":
            suite, preds = read_json(args.suite), read_jsonl(args.predictions)
            if args.task == "relations":
                result = evaluate(suite, preds, model=args.model, representation=args.representation)
            else:
                if args.representation != "final":
                    raise ValueError("event mention extraction accepts spans, not relation passes")
                result = evaluate_event_mentions(suite, preds, model=args.model)
        elif args.command == "verify-history":
            result = verify_many(args.reports)
        elif args.command == "import-history":
            result = import_history(read_json(args.suite), args.report, model=args.model)
        else:
            result = compare([read_json(path) for path in args.reports])
        write_json(args.output, result)
        print(json.dumps({"output": args.output, "status": result.get("status", "written")}
                         if isinstance(result, dict) else {"output": args.output}))
        return 1 if isinstance(result, dict) and result.get("status") == "different" else 0
    except (ValueError, KeyError, OSError, TypeError) as exc:
        parser.exit(2, f"benchmark: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
