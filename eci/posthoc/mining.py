"""The MLflow history: run table, single-parameter ablations, re-run variance, LaTeX table.

Reads the tracking databases with sqlite in read-only mode — never through MlflowClient,
which may try to migrate a database's schema — so agentere's mlflow.db can be mined alongside
workere's without being touched. Runs from both frameworks share param keys (see
config.flatten_params), so an ablation pair can span the two.

What makes two runs "the same experiment" is their params minus the keys that cannot change a
prediction (INFRA) or are implied by another key (DERIVED). Keys introduced after a run was
made are absent from its params; NEUTRAL gives the value that absence stands for (the feature
did not exist, so it was off), so an old run still pairs with a new one that leaves it off.
"""
from __future__ import annotations

import csv
import itertools
import sqlite3
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ABSENT = "\x00absent"

DERIVED = {"prompt.system", "prompt.user_template", "prompt.user_suffix", "prompt.steps",
           "dataset.name", "dataset.repo_id", "dataset.split", "dataset.text_field",
           "dataset.ann_field", "data.labels"}
INFRA = {"experiment.concurrency", "experiment.doc_concurrency", "experiment.retries",
         "experiment.retry_wait", "experiment.timeout", "experiment.tracing",
         "experiment.tracing_name", "llm_cache.enabled", "llm_cache.namespace", "llm_cache.path",
         "few_shot.cot_generation.cache_path", "few_shot.cot_generation.dump_dir",
         "few_shot.human_rationales.dir", "encoder.path", "coref_hint.esl_root",
         "coref_injection.script", "coref_injection.source", "oracle_count.script"}
SEEDS = {"seed", "few_shot.shuffle_pool_seed"}
NEUTRAL = {"coref_hint.enabled": "False", "coref_hint.source": ABSENT,
           "few_shot.cot_generation.negation": "False", "few_shot.repo_id": "",
           "mention_whitelist.enabled": "False", "gold_count.enabled": "False",
           "mention_split.enabled": "False", "syntax.level": "off", "syntax.discourse": "[]",
           "coref_injection.enabled": "False"}
# Sub-keys that mean nothing while their feature is off.
GATED = {"mention_whitelist.": "mention_whitelist.enabled", "coref_hint.": "coref_hint.enabled",
         "mention_split.": "mention_split.enabled", "few_shot.": "few_shot.enabled",
         "few_shot.cot_generation.": "few_shot.cot_generation.enabled",
         "experiment.resampling.": "experiment.resampling.enabled"}
HOLD_EQUAL = {"dataset.max_examples", "active_dataset"}
COARSE = ("active_dataset", "model.default_model_id", "dataset.prompt", "dataset.max_examples",
          "experiment.resampling.n_runs")
METRICS = ("binary_intra_f1", "binary_f1", "micro_f1", "binary_intra_precision",
           "binary_intra_recall", "binary_precision", "binary_recall", "binary_inter_f1", "cost_usd")


@dataclass
class Run:
    uuid: str
    name: str
    status: str
    start: datetime
    db: str
    params: Dict[str, str]
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def short(self) -> str:
        return (self.name or self.uuid).rsplit("_", 1)[-1][:8]

    def value(self, key: str) -> str:
        return self.params.get(key, NEUTRAL.get(key, ABSENT))


def load(dbs: Sequence[Path]) -> List[Run]:
    runs = []
    for db in dbs:
        if not Path(db).exists():
            continue
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        for uuid, name, status, start in con.execute(
                "SELECT run_uuid, name, status, start_time FROM runs WHERE lifecycle_stage='active'"):
            params = dict(con.execute("SELECT key, value FROM params WHERE run_uuid=?", (uuid,)).fetchall())
            metrics = dict(con.execute("SELECT key, value FROM latest_metrics WHERE run_uuid=?", (uuid,)).fetchall())
            runs.append(Run(uuid, name or "", status,
                            datetime.fromtimestamp((start or 0) / 1000, tz=timezone.utc),
                            Path(db).parent.name, params, metrics))
        con.close()
    return sorted(runs, key=lambda r: r.start)


def signature(run: Run, ignore: set, keys: Optional[set] = None) -> Dict[str, str]:
    keys = keys if keys is not None else set(run.params) | set(NEUTRAL)
    sig = {}
    for k in keys:
        if k in ignore:
            continue
        gate = next((g for prefix, g in GATED.items() if k.startswith(prefix) and k != g), None)
        if gate and run.value(gate) == "False":
            continue
        sig[k] = run.value(k)
    return sig


def usable(runs: List[Run], metric: str) -> List[Run]:
    return [r for r in runs if r.status == "FINISHED" and r.params and metric in r.metrics]


def ablations(runs: List[Run], metric: str, param: Optional[str] = None,
              extra_ignore: Sequence[str] = ()) -> List[dict]:
    ignore = INFRA | DERIVED | set(extra_ignore)
    runs = usable(runs, metric)
    keys = set().union(*(set(r.params) for r in runs)) | set(NEUTRAL) if runs else set()
    sigs = [signature(r, ignore, keys) for r in runs]
    out = []
    for (a, sa), (b, sb) in itertools.combinations(zip(runs, sigs), 2):
        diff = [k for k in set(sa) | set(sb) if sa.get(k, ABSENT) != sb.get(k, ABSENT)]
        if len(diff) != 1 or diff[0] in HOLD_EQUAL:
            continue
        k = diff[0]
        if param and k != param:
            continue
        out.append({"param": k, "a": a, "b": b, "value_a": sa.get(k, ABSENT), "value_b": sb.get(k, ABSENT),
                    "metric_a": a.metrics[metric], "metric_b": b.metrics[metric]})
    return sorted(out, key=lambda x: (x["param"], x["a"].start))


def variance(runs: List[Run], metric: str, coarse: bool, keep_seeds: bool = True) -> List[dict]:
    ignore = INFRA | DERIVED | (set() if keep_seeds else SEEDS)
    runs = usable(runs, metric)
    groups: Dict[tuple, List[Run]] = {}
    for r in runs:
        key = tuple(r.value(k) for k in COARSE) if coarse else tuple(sorted(signature(r, ignore).items()))
        groups.setdefault(key, []).append(r)
    out = []
    for key, g in groups.items():
        if len(g) < 2:
            continue
        vals = [r.metrics[metric] for r in g]
        varying = sorted(k for k in set().union(*(set(r.params) for r in g)) - ignore
                         if len({r.value(k) for r in g}) > 1)
        out.append({"runs": g, "n": len(vals), "mean": statistics.fmean(vals),
                    "sd": statistics.stdev(vals), "min": min(vals), "max": max(vals),
                    "range": max(vals) - min(vals), "varying": varying,
                    "label": " ".join(f"{k.split('.')[-1]}={g[0].value(k).split('/')[-1]}" for k in COARSE)})
    return sorted(out, key=lambda x: -x["n"])


def table(runs: List[Run]) -> List[dict]:
    rows = []
    for r in runs:
        p = r.params
        n = p.get("experiment.resampling.n_runs", "") if p.get("experiment.resampling.enabled") == "True" else "1"
        rows.append({
            "date": r.start.strftime("%m-%d %H:%M"), "run": r.short, "db": r.db, "status": r.status[:4],
            "dataset": p.get("active_dataset", ""), "split": p.get("dataset.split", ""),
            "n": p.get("dataset.max_examples", ""), "model": p.get("model.default_model_id", "").split("/")[-1],
            "prompt": p.get("dataset.prompt", ""), "passes": n if p else "",
            "fewshot": p.get("few_shot.n_examples", "") if p.get("few_shot.enabled") == "True" else "-",
            "syntax": p.get("syntax.level", ""), "wl": "wl" if p.get("mention_whitelist.enabled") == "True" else "",
            **{m: r.metrics.get(m) for m in ("binary_f1", "binary_intra_f1", "binary_inter_f1", "cost_usd")},
        })
    return rows


def write_csv(rows: List[dict], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def latex(runs: List[Run], days: int, group: bool) -> str:
    """binary_intra P/R/F1 of every finished run of the last `days`, blocked by dataset, best F1
    bold; with `group`, identical configs collapse to mean +- sd."""
    cutoff = datetime.now(timezone.utc).timestamp() - days * 86400
    runs = [r for r in usable(runs, "binary_intra_f1") if r.start.timestamp() >= cutoff]
    cols = ("model.default_model_id", "dataset.prompt", "dataset.max_examples", "experiment.resampling.n_runs",
            "few_shot.n_examples", "syntax.level")
    by_ds: Dict[str, Dict[tuple, List[Run]]] = {}
    for r in runs:
        key = tuple(r.value(c) for c in cols) if group else (r.short,) + tuple(r.value(c) for c in cols)
        by_ds.setdefault(r.value("active_dataset"), {}).setdefault(key, []).append(r)
    esc = lambda s: str(s).replace("_", r"\_")
    lines = [r"\begin{tabular}{llllll rrr}", r"\toprule",
             r"model & prompt & docs & passes & shots & syntax & P & R & F1 \\", r"\midrule"]
    for ds, groups in sorted(by_ds.items()):
        lines.append(rf"\multicolumn{{9}}{{l}}{{\textbf{{{esc(ds)}}}}} \\")
        stats = []
        for key, g in groups.items():
            vals = {m: [100 * r.metrics[f"binary_intra_{m}"] for r in g] for m in ("precision", "recall", "f1")}
            stats.append((key, g, vals))
        best = max(statistics.fmean(v["f1"]) for _, _, v in stats)
        for key, g, vals in sorted(stats, key=lambda x: -statistics.fmean(x[2]["f1"])):
            cells = []
            for m in ("precision", "recall", "f1"):
                mu = statistics.fmean(vals[m])
                txt = f"{mu:.1f}" + (rf"$\pm${statistics.stdev(vals[m]):.1f}" if len(vals[m]) > 1 else "")
                cells.append(rf"\textbf{{{txt}}}" if m == "f1" and mu == best else txt)
            conf = [g[0].value(c).split("/")[-1] for c in cols]
            lines.append(" & ".join(esc(c) for c in conf) + " & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    return "\n".join(lines)
