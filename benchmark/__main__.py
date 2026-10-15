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
    # New execution commands are lazy-imported: offline scoring never loads a model registry.
    for command in ("run", "batch", "sweep"):
        p = commands.add_parser(command, help="execute experiment YAML, campaign YAML, or Cartesian sweep")
        p.add_argument("configs", nargs="+" if command == "run" else 1)
        p.add_argument("--output-root")
        p.add_argument("--output", required=True, help="campaign receipt JSON")
        p.add_argument("--dry-run", action="store_true")
    p = commands.add_parser("reevaluate", help="score stored universal predictions without loading models")
    p.add_argument("run_dir")
    p.add_argument("--output", required=True)
    p.add_argument("--samples", type=int)
    p.add_argument("--confidence", type=float)
    p.add_argument("--seed", type=int)
    p = commands.add_parser("plugins", help="list registered versioned component implementations")
    p.add_argument("--module", action="append", default=[])
    p = commands.add_parser("compare-runs", help="compare framework runs and imported historical baselines")
    p.add_argument("runs", nargs="+")
    p.add_argument("--metric", default="binary")
    p.add_argument("--samples", type=int, default=1000)
    p.add_argument("--confidence", type=float, default=.95)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)
    p.add_argument("--markdown")
    args = parser.parse_args(argv)
    try:
        if args.command in {"run", "batch", "sweep"}:
            from benchmark.campaign import execute_many
            from benchmark.configuration import expand_campaign, expand_sweep, load_experiment, seed_configs
            if args.command == "run":
                configs = [c for path in args.configs for c in seed_configs(load_experiment(path))]
            else:
                configs = list((expand_campaign if args.command == "batch" else expand_sweep)(args.configs[0]))
            result = execute_many(configs, output_root=args.output_root, output=args.output if not args.dry_run else None,
                                  dry_run=args.dry_run)
            if args.dry_run:
                write_json(args.output, result)
            print(json.dumps(result))
            return 1 if result.get("status") == "errors" else 0
        if args.command == "reevaluate":
            from benchmark.engine import reevaluate
            reevaluate(args.run_dir, args.output, samples=args.samples, confidence=args.confidence, seed=args.seed)
            return 0
        if args.command == "plugins":
            from benchmark.registry import load_plugins
            print(json.dumps(load_plugins(args.module).describe(), indent=2))
            return 0
        if args.command == "compare-runs":
            from benchmark.comparison import compare_runs, markdown_table
            result = compare_runs(args.runs, metric=args.metric, samples=args.samples, confidence=args.confidence, seed=args.seed)
            write_json(args.output, result)
            if args.markdown:
                from pathlib import Path
                with Path(args.markdown).open("x", encoding="utf-8") as stream:
                    stream.write(markdown_table(result))
            return 0
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
