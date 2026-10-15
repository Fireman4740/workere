"""Commands for zero-cost planning, pipeline execution and offline re-evaluation."""
import argparse
import json

from benchmark.configuration import load_experiment, seed_configs
from benchmark.io import write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("dry-run")
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("run")
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--output-root")
    p.add_argument("--mock", action="store_true")
    args = parser.parse_args(argv)
    config = load_experiment(args.config)
    if args.command == "dry-run":
        from pipelines.jev.planning import plan
        result = plan(config)
        write_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 1 if result["context_overflows"] else 0
    if args.mock:
        config.data["pipeline"]["relation_predictor"]["params"]["client"]["mode"] = "mock"
        config.data["execution"].update(allow_network=False, allow_paid=False)
        config.data["name"] += "-MOCK"
    from benchmark.campaign import execute_many
    result = execute_many(seed_configs(config), output_root=args.output_root, output=args.output)
    for run in result["runs"]:
        if run["status"] in {"completed", "completed_with_errors"}:
            from benchmark.io import read_json
            from pathlib import Path
            directory = Path(run["run_dir"])
            metrics = read_json(directory / "metrics.json")
            print(json.dumps({"run_dir": run["run_dir"], "summary": metrics["summary"],
                              "uncertainty": metrics["uncertainty"],
                              "metadata": read_json(directory / "metadata.json")}, ensure_ascii=False), flush=True)
    return 1 if result["status"] == "errors" else 0


if __name__ == "__main__":
    raise SystemExit(main())
