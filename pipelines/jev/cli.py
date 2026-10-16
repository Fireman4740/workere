"""Commands for zero-cost planning, pipeline execution and offline re-evaluation."""
import argparse
import json

from benchmark.configuration import expand_campaign, load_experiment, read_yaml, seed_configs
from benchmark.io import write_json


def _configs(path):
    """A single experiment or a `workere-campaign/v1` list, expanded to effective configs."""
    if read_yaml(path).get("schema") == "workere-campaign/v1":
        return list(expand_campaign(path))
    return list(seed_configs(load_experiment(path)))


def _execute_parallel(configs, *, output_root=None, output=None):
    """Campaign entries are independent runs; execute them concurrently, one directory each."""
    from concurrent.futures import ThreadPoolExecutor
    from benchmark.campaign import execute_many
    if len(configs) == 1:
        return execute_many(configs, output_root=output_root, output=output)
    with ThreadPoolExecutor(max_workers=len(configs)) as pool:
        parts = list(pool.map(lambda c: execute_many([c], output_root=output_root), configs))
    runs = [run for part in parts for run in part["runs"]]
    result = {"schema": "workere-campaign-result/v1", "runs": runs,
              "status": "completed" if all(r["status"] == "completed" for r in runs) else "errors"}
    if output:
        write_json(output, result)
    return result


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
    p = sub.add_parser("analyze")
    p.add_argument("run_dir")
    p.add_argument("--output-dir", required=True)
    p = sub.add_parser("rethreshold")
    p.add_argument("run_dir")
    p.add_argument("--threshold", type=float, required=True)
    p.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    if args.command in {"analyze", "rethreshold"}:
        from pipelines.jev.analysis import analyze, rethreshold
        result = (analyze(args.run_dir, args.output_dir) if args.command == "analyze" else
                  rethreshold(args.run_dir, args.threshold, args.output_dir))
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 0
    configs = _configs(args.config)
    if args.command == "dry-run":
        from pipelines.jev.planning import plan
        plans = [plan(config) for config in configs]
        result = plans[0] if len(plans) == 1 else {"schema": "workere-jev-campaign-plan/v1", "plans": plans,
            "calls_without_cache_or_retries": sum(p["calls_without_cache_or_retries"] for p in plans),
            "estimated_input_cost_usd_at_configured_rate": (sum(p["estimated_input_cost_usd_at_configured_rate"] for p in plans)
                if all(p["estimated_input_cost_usd_at_configured_rate"] is not None for p in plans) else None),
            "context_overflows": [o for p in plans for o in p["context_overflows"]]}
        write_json(args.output, result)
        print(json.dumps({k: v for k, v in result.items() if k != "plans"} if len(plans) > 1 else result,
                         ensure_ascii=False)[:20000], flush=True)
        return 1 if result["context_overflows"] else 0
    if args.mock:
        for config in configs:
            config.data["pipeline"]["relation_predictor"]["params"]["client"]["mode"] = "mock"
            config.data["execution"].update(allow_network=False, allow_paid=False)
            config.data["name"] += "-MOCK"
    result = _execute_parallel(configs, output_root=args.output_root, output=args.output)
    for run in result["runs"]:
        if run["status"] == "failed":
            print(json.dumps(run, ensure_ascii=False), flush=True)
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
