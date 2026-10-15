"""Sequential bounded orchestration; every effective configuration has its own run directory."""
from benchmark.engine import ExecutionError, execute
from benchmark.io import write_json


def execute_many(configs, *, output_root=None, output=None, dry_run=False):
    configs = list(configs)
    if not configs:
        raise ValueError("campaign contains no runs")
    if dry_run:
        return {"schema": "workere-campaign-plan/v1", "runs": [cfg.data for cfg in configs], "count": len(configs)}
    runs = []
    for config in configs:
        try:
            run = execute(config, output_root=output_root)
        except ExecutionError as exc:
            run = {"run_dir": str(exc.run_dir.resolve()), "status": "failed", "error": str(exc)}
        runs.append(run)
        print(f"{config.data['name']} seed={config.data['seed']} status={run['status']} run={run['run_dir']}", flush=True)
    result = {"schema": "workere-campaign-result/v1", "runs": runs,
              "status": "completed" if all(r["status"] == "completed" for r in runs) else "errors"}
    if output:
        write_json(output, result)
    return result
