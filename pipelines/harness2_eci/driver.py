"""Run a Harness² sweep over ESC documents, then score every rollout against annotations.

Config schema `workere-harness2/v1` (see experiments/jev/harness2/*.yaml). Annotations are
read only after the sweep finished, from the hidden store, for the base, every probe and the
selected final; selection itself is label-free.
"""
import json
import os
import random
from pathlib import Path

from benchmark.configuration import read_yaml
from benchmark.io import write_json

SCHEMA = "workere-harness2/v1"
LABELS = ("base1", "inv1", "dom1", "agg1")


def _prf(c):
    p = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
    r = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0, **c}


def _micro(rows, label, key="binary"):
    total = {"tp": 0, "fp": 0, "fn": 0}
    for row in rows:
        for k in total:
            total[k] += row["scores"][label][key][k]
    return _prf(total)


def paired_bootstrap(rows, a, b, samples=2000, seed=42):
    """Document-resampled micro-F1 difference b - a, in points."""
    rng, deltas = random.Random(seed), []
    for _ in range(samples):
        draw = [rng.choice(rows) for _ in rows]
        deltas.append(100 * (_micro(draw, b)["f1"] - _micro(draw, a)["f1"]))
    deltas.sort()
    return {"delta_points": 100 * (_micro(rows, b)["f1"] - _micro(rows, a)["f1"]),
            "ci95": [deltas[int(.025 * samples)], deltas[int(.975 * samples) - 1]],
            "p_gain": sum(d > 0 for d in deltas) / samples, "samples": samples, "seed": seed,
            "documents": len(rows)}


def _replace(path, value):
    """The work root persists across repair runs; its summaries are rewritten, not appended."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding="utf-8")


def plan(cfg):
    docs = cfg["documents"]
    return {"schema": "workere-harness2-plan/v1", "executor": cfg["executor"], "documents": len(docs),
            "topics": sorted({d.split("_")[0] for d in docs}, key=int), "k": cfg["k"], "modes": cfg["modes"],
            "proposer_model": cfg["proposer"]["model"], "budget": cfg["budget"],
            "proposer_sessions_upper_bound": len(docs) * len(cfg["modes"]) * (2 * cfg["k"] + cfg["k"] + 2),
            "executor_rollouts_upper_bound": len(docs) * (cfg["k"] + len(cfg["modes"]) * 3 * cfg["k"])}


def run(config_path, output):
    cfg = read_yaml(config_path)
    if cfg.get("schema") != SCHEMA:
        raise ValueError(f"expected schema {SCHEMA}")
    model = cfg["proposer"]["model"]
    # harness2.config reads these once at import; they name the improver, never a credential.
    os.environ["HARNESS2_IMPROVER_MODEL"] = os.environ["HARNESS2_SOLVER_MODEL"] = model
    root = Path(cfg["work_root"]).expanduser() / f"{cfg['name']}-v{cfg['code_version']}"
    os.environ["HARNESS2_RUNS_DIR"] = str(root / "runs")
    from harness2.core import schedule
    from pipelines.harness2_eci.adapter import EciAdapter
    from pipelines.harness2_eci.proposer import OpenRouterProposer
    from pipelines.jev.spend import SharedBudget

    from pipelines.harness2_eci.proposer import openrouter_key
    openrouter_key()  # fail before any work when the credential is absent
    budget = SharedBudget(**cfg["budget"])
    print("H2_PLAN " + json.dumps(plan(cfg)), flush=True)
    print("PLAN_BUDGET " + json.dumps(budget.status()), flush=True)
    base = Path(config_path).resolve().parent
    adapter = EciAdapter(executor=cfg["executor"], template_path=base / cfg["executor_template"],
                         suites_dir=base / cfg["suites_dir"], documents=cfg["documents"], root=root, model=model)
    prop = OpenRouterProposer(adapter.spec.channels["proposer"], audit_dir=root / "audit", budget=budget,
                              session_reserve_usd=cfg["proposer"]["session_reserve_usd"])
    sweep = schedule.run_sweep(cfg["k"], adapter=adapter, prop=prop, runs_root=adapter.spec.runs_root,
                               library_root=root / "library", modes=cfg["modes"], seed=cfg["seed"],
                               only_tasks=cfg["documents"], domain_parallel=cfg["domain_parallel"])
    _replace(root / "sweep.json", sweep)
    result = score(adapter, cfg, sweep)
    result["budget"] = budget.status()
    write_json(output, result)
    _replace(root / "result.json", result)
    print("H2_RESULT " + json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["documents_scored"] == len(cfg["documents"]) else 1


def score(adapter, cfg, sweep):
    from harness2.core.schedule import Namespace
    rows, missing = [], []
    for task in cfg["documents"]:
        row = {"document": task, "topic": adapter.domain_of(task), "scores": {}, "selected": {}}
        for mode in cfg["modes"]:
            n = Namespace(adapter.spec.name, adapter.substrate.name, mode, cfg["k"], cfg["seed"], "")
            summary_path = n.task_dir(adapter.spec.runs_root, task) / "summary.json"
            summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
            sel = summary.get("selections", {}).get(mode, {})
            row["selected"][mode] = {"label": sel.get("chosen_label"), "is_base": sel.get("chosen_tree_is_base"),
                                     "stages": summary.get("modes", {}).get(mode, {}).get("stages", [])}
            for label in LABELS:
                run_ns = Namespace(adapter.spec.name, adapter.substrate.name, "agg", cfg["k"], cfg["seed"], "") \
                    if label.startswith("base") else n
                run_dir = adapter.root / "runs" / "rollouts" / run_ns.run_id(task, label)
                got = adapter.score(run_dir, task) if run_dir.is_dir() else None
                key = label if label.startswith("base") else f"{mode}:{label}"
                if got is not None:
                    row["scores"][key] = got
            chosen = sel.get("chosen_label")
            if chosen and f"{mode}:{chosen}" in row["scores"]:
                row["scores"][f"{mode}:selected"] = row["scores"][f"{mode}:{chosen}"]
        needed = ["base1"] + [f"{m}:selected" for m in cfg["modes"]]
        (rows if all(k in row["scores"] for k in needed) else missing).append(row)
    out = {"schema": "workere-harness2-result/v1", "executor": cfg["executor"], "k": cfg["k"],
           "documents_scored": len(rows), "documents_missing": [r["document"] for r in missing],
           "per_document": rows + missing, "micro": {}, "paired_vs_base": {}}
    if rows:
        keys = sorted({k for r in rows for k in r["scores"]})
        common = [k for k in keys if all(k in r["scores"] for r in rows)]
        for key in common:
            out["micro"][key] = {g: _prf({c: sum(r["scores"][key][g][c] for r in rows) for c in ("tp", "fp", "fn")})
                                 for g in ("binary", "binary_intra", "binary_inter")}
        for key in common:
            if key != "base1":
                out["paired_vs_base"][key] = paired_bootstrap(rows, "base1", key)
    return out
