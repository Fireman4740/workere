#!/usr/bin/env python3
"""Post-hoc analysis of finished runs — workere's and agentere's. No model calls.

A RUN is a report .json, a workere run directory, a run-id fragment (e.g. 85dccc90) searched in
workere's logs/runs and agentere's logs/allatonce, or an MLflow run-uuid prefix.

  sweep RUN [RUN ...]    vote-threshold sweep; --coref (ESC: ECB+ chains, MAVEN: span dataset,
                         or --chains surface) and --closure add the post-hoc variants;
                         --undirected scores unordered pairs; --common-docs aligns runs
  rescore RUN            vote_to_binary x {binary_default, novote_norel}, OR / AND direction
                         rules, and a recompute of the logged metrics from the rows
  ranking RUN [RUN ...]  votes as a cause-retrieval ranker vs TF-IDF / embedding / random
  runs                   the MLflow history as a table (--csv / --xlsx)
  ablations              run pairs differing in exactly one param (--param, --metric)
  variance               re-run spread per config (--coarse, --ignore-seeds)
  latex                  LaTeX table of binary_intra P/R/F1 (--days, --group)
  graph RUN DOC_ID       TP/FP/FN causal graph of one document (SVG)
  traces RUN             pretty-print trace lines (--doc, --role, --limit, --full)
  spend RUN              token spend per document against its mention count
  hist RUN               the per-run histogram set, into --out
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eci.config import ROOT  # noqa: E402
from eci.posthoc import runs as RUNS  # noqa: E402


def cmd_sweep(a):
    from eci.posthoc import sweep as S
    loaded = [RUNS.load(s) for s in a.runs]
    keep = RUNS.restrict(loaded) if a.common_docs and len(loaded) > 1 else None
    if keep is not None:
        print(f"restricted to the {len(keep)} documents all {len(loaded)} runs cover\n")
    variants = ["base"]
    if a.coref:
        variants += ["+coref", "+coref_edges"]
    if a.closure:
        variants += ["+trans", "+trans1", "+trans1(unanim)"] + (["+coref+trans"] if a.coref else [])
    payload = {}
    for run in loaded:
        chains = None
        if a.coref or a.closure:
            run.ensure_sentences()
        if a.coref:
            source = a.chains or S.default_chain_source(run)
            if source is None:
                raise SystemExit(f"{run.name}: no chain source for this dataset; pass --chains surface")
            chains = S.chains_for(run, source, a.esl_root)
            covered = sum(1 for c in chains.values() if c)
            print(f"{run.name}: coreference chains ({source}) for {covered}/{len(run.docs)} documents")
        res = S.sweep(run, variants, chains, keep, a.undirected, a.drop_in_chain)
        label = f"{run.name}  ({run.dataset}, n_runs={run.n_runs}, {len(keep or run.docs)} docs, " \
                f"{'undirected' if a.undirected else 'ordered'} pairs)"
        print(S.format_table(run, res, variants, label))
        if keep is None and not a.undirected and (line := S.sanity_line(run, res)):
            print(line)
        print()
        payload[run.name] = {str(t): {v: {s: dict(zip(("P", "R", "F1", "tp", "fp", "fn"), row[v][s]))
                                          for s in S.SCOPES} for v in variants} for t, row in res.items()}
    if a.json:
        Path(a.json).write_text(json.dumps(payload, indent=1))
        print(f"wrote {a.json}")


def cmd_rescore(a):
    from eci.posthoc import rescore
    run = RUNS.load(a.run)
    rc = rescore.recompute(run)
    print(f"=== {run.name}: logged vs recomputed from rows")
    for k in ("micro_f1", "macro_f1"):
        print(f"  {k:14s} {rc['logged'][k]:.4f}  {rc['recomputed'][k]:.4f}")
    for k in ("binary", "binary_intra", "binary_inter"):
        o, n = rc["logged"][k], rc["recomputed"][k]
        print(f"  {k:14s} {o['f1']:.4f}  {n['f1']:.4f}   (P {n['precision']:.4f} R {n['recall']:.4f})")
    print(f"\n=== decision rules on unordered pairs  ALL / INTRA / INTER  P/R/F1")
    for name, m in rescore.binary_rules(run).items():
        print(f"  {name:44s}" + "".join(f"  {m[s][0]:5.1f}/{m[s][1]:5.1f}/{m[s][2]:5.1f}" for s in ("all", "intra", "inter")))


def cmd_ranking(a):
    import yaml
    from eci.posthoc import ranking
    loaded = [RUNS.load(s) for s in a.runs]
    model = a.embedding or yaml.safe_load((ROOT / "config.yaml").read_text())["few_shot"]["bert_model"]
    res = ranking.rank(loaded, a.k_max, a.seed, model)
    for setting, r in res.items():
        first = next(iter(r["methods"].values()))
        print(f"\n{setting}: {r['n_docs']} docs, {first['n_queries']} queries, {first['mean_items']:.1f} items/query")
        print(f"{'method':46}" + "".join(f"{'H@' + str(k):>6}" for k in range(1, a.k_max + 1)) + f"{'MRR':>6}")
        for name, m in r["methods"].items():
            print(f"{name[:46]:46}" + "".join(f"{h:6.1f}" for h in m["hit"]) + f"{m['mrr']:6.1f}")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))


def _mlflow_runs(a):
    from eci.posthoc import mining
    return mining, mining.load([Path(d) for d in (a.db or RUNS.TRACKING_DBS)])


def cmd_runs(a):
    mining, runs = _mlflow_runs(a)
    rows = mining.table(runs if a.all else [r for r in runs if r.status == "FINISHED" and r.params])
    if a.csv:
        mining.write_csv(rows, a.csv)
    if a.xlsx:
        import pandas as pd
        pd.DataFrame(rows).to_excel(a.xlsx, index=False)
    fmt = lambda v: "" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v)
    keys = list(rows[0]) if rows else []
    widths = {k: max(len(k), *(len(fmt(r[k])) for r in rows)) for k in keys}
    print("  ".join(k.ljust(widths[k]) for k in keys))
    for r in rows:
        print("  ".join(fmt(r[k]).ljust(widths[k]) for k in keys))


def cmd_ablations(a):
    mining, runs = _mlflow_runs(a)
    pairs = mining.ablations(runs, a.metric, a.param, a.ignore or ())
    print(f"{len(pairs)} single-param ablation pairs on {a.metric}")
    for p in pairs:
        va, vb = p["value_a"][:40], p["value_b"][:40]
        print(f"  {p['param']:40s} {p['a'].short} {va!r:>30} {p['metric_a']:.4f}  |  "
              f"{p['b'].short} {vb!r:>30} {p['metric_b']:.4f}  d={p['metric_b'] - p['metric_a']:+.4f}")


def cmd_variance(a):
    mining, runs = _mlflow_runs(a)
    groups = mining.variance(runs, a.metric, a.coarse, not a.ignore_seeds)
    for g in groups:
        print(f"n={g['n']:2d} mean={g['mean']:.4f} sd={g['sd']:.4f} range={g['range']:.4f}  {g['label']}")
        print(f"      runs: {' '.join(r.short for r in g['runs'])}" + (f"   varying: {', '.join(g['varying'])}" if g["varying"] else ""))


def cmd_latex(a):
    mining, runs = _mlflow_runs(a)
    print(mining.latex(runs, a.days, a.group))


def cmd_graph(a):
    from eci import graphs
    run = RUNS.load(a.run)
    run.ensure_sentences()
    doc = run.docs.get(a.doc)
    if doc is None:
        raise SystemExit(f"{a.doc} is not in {run.name}; documents: {', '.join(list(run.docs)[:10])} ...")
    rows = [r for r in run.results["per_pair_predictions"] if r["id"] == a.doc]
    gold = [(r["pair"].split(",")[0], r["gold"], r["pair"].split(",")[1]) for r in rows if r["gold"] != "norel"]
    pred = [(r["pair"].split(",")[0], r["pred"], r["pair"].split(",")[1]) for r in rows if r["pred"] != "norel"]
    path = graphs.render_graph(Path(a.out), a.doc, doc.surfaces, gold, pred)
    print(path)


def cmd_traces(a):
    from eci.posthoc import traces
    run = RUNS.load(a.run)
    print(traces.show(run.path.with_name(run.name + ".traces.jsonl"), a.doc, a.role, a.limit, a.width,
                      a.full, not a.no_color))


def cmd_spend(a):
    from eci.posthoc import traces
    run = RUNS.load(a.run)
    run.ensure_sentences()
    per = traces.spend_per_doc(run.path.with_name(run.name + ".traces.jsonl"))
    rows = sorted((len(run.docs[d].sentence_of), d, v) for d, v in per.items() if d in run.docs)
    print(f"{'mentions':>8}  {'calls':>5}  {'in':>9}  {'out':>9}  {'reason':>9}  {'cost':>8}  doc")
    for m, d, v in rows:
        print(f"{m:8d}  {v['calls']:5d}  {v['in']:9d}  {v['out']:9d}  {v['reasoning']:9d}  {v['cost']:8.4f}  {d}")
    if len(rows) > 2:
        import numpy as np
        x = np.array([r[0] for r in rows], float)
        y = np.array([r[2]["out"] for r in rows], float)
        slope, icpt = np.polyfit(x, y, 1)
        print(f"\noutput tokens ~ {slope:.1f} x mentions + {icpt:.0f}   (r = {np.corrcoef(x, y)[0, 1]:.2f})")


def cmd_hist(a):
    from eci import graphs
    run = RUNS.load(a.run)
    if "doc_index" not in run.results:
        raise SystemExit("hist needs a workere report (agentere reports carry no word positions)")
    for p in graphs.histograms(run.results, run.dataset, Path(a.out), f"{run.name}."):
        print(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("sweep")
    p.add_argument("runs", nargs="+")
    p.add_argument("--coref", action="store_true")
    p.add_argument("--chains", choices=["esc_ecb", "maven_span", "surface"])
    p.add_argument("--esl-root", default=str(RUNS.AGENTERE / "data" / "EventStoryLine"))
    p.add_argument("--drop-in-chain", action="store_true")
    p.add_argument("--closure", action="store_true")
    p.add_argument("--undirected", action="store_true")
    p.add_argument("--common-docs", action="store_true")
    p.add_argument("--json")
    p.set_defaults(fn=cmd_sweep)

    p = sub.add_parser("rescore")
    p.add_argument("run")
    p.set_defaults(fn=cmd_rescore)

    p = sub.add_parser("ranking")
    p.add_argument("runs", nargs="+")
    p.add_argument("--k-max", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--embedding")
    p.add_argument("--json")
    p.set_defaults(fn=cmd_ranking)

    for name, fn in (("runs", cmd_runs), ("ablations", cmd_ablations), ("variance", cmd_variance),
                     ("latex", cmd_latex)):
        p = sub.add_parser(name)
        p.add_argument("--db", action="append", help="tracking sqlite file(s); default: workere's and agentere's")
        if name == "runs":
            p.add_argument("--all", action="store_true", help="include failed / param-less runs")
            p.add_argument("--csv")
            p.add_argument("--xlsx")
        if name in ("ablations", "variance"):
            p.add_argument("--metric", default="binary_intra_f1")
        if name == "ablations":
            p.add_argument("--param")
            p.add_argument("--ignore", nargs="*")
        if name == "variance":
            p.add_argument("--coarse", action="store_true")
            p.add_argument("--ignore-seeds", action="store_true")
        if name == "latex":
            p.add_argument("--days", type=int, default=30)
            p.add_argument("--group", action="store_true")
        p.set_defaults(fn=fn)

    p = sub.add_parser("graph")
    p.add_argument("run")
    p.add_argument("doc")
    p.add_argument("--out", default="graphs")
    p.set_defaults(fn=cmd_graph)

    p = sub.add_parser("traces")
    p.add_argument("run")
    p.add_argument("--doc")
    p.add_argument("--role")
    p.add_argument("--limit", type=int, default=3)
    p.add_argument("--width", type=int, default=110)
    p.add_argument("--full", action="store_true")
    p.add_argument("--no-color", action="store_true")
    p.set_defaults(fn=cmd_traces)

    p = sub.add_parser("spend")
    p.add_argument("run")
    p.set_defaults(fn=cmd_spend)

    p = sub.add_parser("hist")
    p.add_argument("run")
    p.add_argument("--out", default="figures")
    p.set_defaults(fn=cmd_hist)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
