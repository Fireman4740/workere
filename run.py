#!/usr/bin/env python3
"""Run an ECI experiment.

  python run.py                                     # config.yaml as it is
  python run.py --set experiment.resampling.n_runs=5 --set datasets.maven_ere.max_examples=20
  python run.py --queue queues/example.yaml [--only name1 name2]
  python run.py --resume logs/runs/<stem> [--retry-failed]
  python run.py --from-agentere ../agentere/logs/allatonce/run_<...>.config.yaml
  python run.py --dry-run                           # plan only: prompts, demos, call count; no calls

Needs OPENROUTER_API_KEY in the environment (except --dry-run).
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eci.config import DEFAULT_CONFIG, ConfigError, load_config, n_passes, overrides_from_agentere  # noqa: E402


def parse_set(items):
    """`a.b.c=value` -> nested dict, value parsed as YAML (numbers, booleans, lists, null)."""
    out: dict = {}
    for item in items or []:
        key, _, raw = item.partition("=")
        if not _:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return out


def merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def dry_run(cfg: dict) -> None:
    """Build every prompt and the few-shot plan without calling any model."""
    from eci.data import load_eval_docs
    from eci.fewshot import FewShotBank
    from eci.runner import Runner
    from eci.tracking import RunDir
    import tempfile

    runner = Runner(cfg, RunDir(Path(tempfile.mkdtemp()) / "dry_run"), factory=lambda *a: None)
    runner.cfg["mlflow"]["enabled"] = False
    docs = load_eval_docs(cfg)
    runner.decorate(docs)
    n = n_passes(cfg)
    demos_needed = 0
    if cfg["few_shot"]["enabled"]:
        bank = FewShotBank(cfg, runner.prompt)
        bank.load(runner.decorate)
        bank.build_plan(docs)
        bank.load_cache()
        demos_needed = len(bank.missing_cots())
        runner.bank = bank
    chars = 0
    for d in docs:
        msgs_len = len(runner._user_prompt(d, d.pair_list, None, None, None))
        chars += msgs_len
    blocks = 3 if cfg["mention_split"]["enabled"] else 1
    calls = len(docs) * n * blocks * ((len(runner.prompt.steps) + 1) if runner.prompt.steps else 1)
    print(f"{len(docs)} documents x {n} passes x {blocks} block(s) = {calls} inference calls")
    print(f"test prompts: {chars:,} chars (~{chars // 4:,} tokens) per pass, demos excluded")
    if cfg["few_shot"]["enabled"]:
        print(f"CoT synthesis needed for {demos_needed} demo documents (the rest are cached)")
    first = docs[0]
    print("\n===== first document, pass 0 user message =====\n")
    print(runner._user_prompt(first, first.pair_list, None, None, None)[:3000])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--set", action="append", metavar="KEY=VALUE", help="override one config key")
    ap.add_argument("--queue", help="queue file: experiments: [{name, overrides}]")
    ap.add_argument("--only", nargs="+", metavar="NAME")
    ap.add_argument("--resume", metavar="RUN_DIR")
    ap.add_argument("--retry-failed", action="store_true", help="with --resume: rerun failed documents")
    ap.add_argument("--from-agentere", metavar="CONFIG_YAML", help="re-run an agentere run's config snapshot")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from eci import runner
    try:
        if args.resume:
            runner.resume(args.resume, args.retry_failed)
            return
        if args.queue:
            sys.exit(1 if runner.run_queue(args.queue, args.only, args.config) else 0)
        overrides = parse_set(args.set)
        if args.from_agentere:
            base = yaml.safe_load(open(args.config, encoding="utf-8"))
            snap = yaml.safe_load(open(args.from_agentere, encoding="utf-8"))
            imported, dropped = overrides_from_agentere(snap, base)
            if dropped:
                print(f"agentere keys not declared here (ignored): {', '.join(sorted(dropped))}")
            overrides = merge(imported, overrides)
        cfg = load_config(args.config, overrides)
    except ConfigError as e:
        raise SystemExit(f"config error: {e}")
    if args.dry_run:
        dry_run(cfg)
        return
    runner.run_config(cfg)


if __name__ == "__main__":
    main()
