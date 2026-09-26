"""Locate a finished run and load it as per-document vote tables.

A run is named by any of: a report `.json` path, a workere run directory, an 8-hex run id or
any fragment of the run name (searched in workere's and agentere's log directories), or an
MLflow run-uuid prefix (looked up read-only in the tracking databases). agentere reports load
too — the report schema is shared — except that they lack `doc_index`, so mention -> sentence
maps come from the dataset instead, loaded through the run's own config.
"""
from __future__ import annotations

import json
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from eci.config import ROOT
from eci.labels import CAUSAL, is_norel

Pair = Tuple[str, str]

AGENTERE = ROOT.parent / "agentere"
SEARCH_DIRS = [ROOT / "logs" / "runs", AGENTERE / "logs" / "allatonce"]
TRACKING_DBS = [ROOT / "mlflow.db", AGENTERE / "mlflow.db"]


@dataclass
class DocVotes:
    id: str
    gold: Dict[Pair, str]                  # ordered pair -> causal gold label
    votes: Dict[Pair, Counter]             # ordered pair -> Counter(causal label -> passes)
    all_votes: Dict[Pair, dict]            # ordered pair -> full vote_counts (norel included)
    row_rel: Dict[Pair, str]               # ordered pair -> intra/inter/unknown, from the rows
    sentence_of: Dict[str, int] = field(default_factory=dict)
    surfaces: Dict[str, str] = field(default_factory=dict)
    first_token: Dict[str, int] = field(default_factory=dict)

    def rel(self, a: str, b: str) -> str:
        if self.sentence_of:
            sa, sb = self.sentence_of.get(a), self.sentence_of.get(b)
            if sa is None or sb is None:
                return "unknown"
            return "intra" if sa == sb else "inter"
        return self.row_rel.get((a, b), "unknown")

    def causal_votes(self, pair: Pair) -> int:
        return sum(self.votes.get(pair, {}).values())


@dataclass
class RunVotes:
    name: str
    path: Path
    n_runs: int
    cfg: dict
    results: dict
    docs: Dict[str, DocVotes]
    framework: str

    @property
    def dataset(self) -> str:
        return self.cfg["active_dataset"]

    def eval_docs(self):
        """The run's documents as `Doc`s (dataset reload through the run's config)."""
        from eci.data import load_split
        ds = self.cfg["datasets"][self.dataset]
        binary = self.cfg["data"]["binary_undirected"]
        docs = load_split(ds["repo_id"], ds["split"], text_field=ds["text_field"],
                          ann_field=ds["ann_field"], valid_labels=None, binary_undirected=binary,
                          shuffle_pair_list=False)
        return {d.id: d for d in docs if d.id in self.docs}

    def ensure_sentences(self) -> None:
        """Fill mention -> sentence maps (needed for pairs no row mentions)."""
        missing = [d for d in self.docs.values() if not d.sentence_of]
        if not missing:
            return
        by_id = self.eval_docs()
        for d in missing:
            src = by_id.get(d.id)
            if src is not None:
                d.sentence_of, d.surfaces, d.first_token = src.sentence_of, src.surfaces, src.first_token


def find_report(selector: str) -> Path:
    p = Path(selector)
    if p.is_file() and p.suffix == ".json":
        return p
    if p.is_dir():
        hits = [f for f in p.glob("run_*.json")]
        if len(hits) == 1:
            return hits[0]
    hits = []
    for d in SEARCH_DIRS:
        if d.is_dir():
            hits += [f for f in d.rglob(f"run_*{selector}*.json") if not f.name.endswith(".traces.json")]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise SystemExit(f"'{selector}' is ambiguous:\n" + "\n".join(f"  {h}" for h in hits[:10]))
    for db in TRACKING_DBS:     # an MLflow run-uuid prefix
        if db.exists():
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            rows = con.execute("SELECT name FROM runs WHERE run_uuid LIKE ?", (selector + "%",)).fetchall()
            con.close()
            if len(rows) == 1 and rows[0][0]:
                return find_report(rows[0][0].rsplit("_", 1)[-1])
    raise SystemExit(f"no run report matches '{selector}'")


def load(selector: str) -> RunVotes:
    path = find_report(selector)
    payload = json.loads(path.read_text(encoding="utf-8"))
    res, cfg = payload["results"], payload["config"]
    rs = cfg["experiment"]["resampling"]
    n_runs = rs["n_runs"] if rs["enabled"] else 1
    docs: Dict[str, DocVotes] = {m["id"]: DocVotes(m["id"], {}, {}, {}, {}) for m in res["per_doc_metrics"]}
    for row in res["per_pair_predictions"]:
        a, b = (x.strip() for x in row["pair"].split(",", 1))
        d = docs.setdefault(row["id"], DocVotes(row["id"], {}, {}, {}, {}))
        vc = row.get("vote_counts") or {}
        if not vc and not is_norel(row["pred"]):
            vc = {row["pred"]: n_runs}   # an unvoted-but-predicted pair: unanimous
        causal = Counter({k: v for k, v in vc.items() if k in CAUSAL})
        if causal:
            d.votes[(a, b)] = causal
        d.all_votes[(a, b)] = vc
        if row["gold"] in CAUSAL:
            d.gold[(a, b)] = row["gold"]
        d.row_rel[(a, b)] = row.get("sentence_relation", "unknown")
    for doc_id, idx in (res.get("doc_index") or {}).items():
        if doc_id in docs:
            docs[doc_id].sentence_of = idx["mention_sentence"]
            docs[doc_id].surfaces = idx["surfaces"]
            docs[doc_id].first_token = idx["first_token"]
    return RunVotes(name=path.stem, path=path, n_runs=n_runs, cfg=cfg, results=res, docs=docs,
                    framework=payload.get("framework", "agentere"))


def restrict(runs: List[RunVotes]) -> Set[str]:
    """Documents every run covers."""
    return set.intersection(*(set(r.docs) for r in runs))
