"""Event-coreference chains, keyed by mention id of the standard-eci documents.

Three sources, one shape — {mention_id: chain}, where a chain is the sorted tuple of the
coreferent mention ids present in the document (size >= 2; singletons say nothing):

  esc_ecb     ESC documents. The standard-eci parquet dropped the chains after using them to
              propagate the gold, so they are recovered from the source corpora: ESC's
              annotated_data (ESC mention ids) joined to ECB+'s CROSS_DOC/INTRA_DOC_COREF on
              the token span, because ESC renumbered the mention ids. 99.9% of ESC action
              spans occur verbatim as an ECB+ action span.
  maven_span  MAVEN-ERE documents. MAVEN's own coreference layer survives in
              Nofing/Maven-ERE-span; mentions are joined on token span and linked by
              union-find. Re-expanding that dataset's CAUSE+PRECONDITION through these chains
              reproduces the published gold exactly on all dev and test documents.
  surface     any dataset: mentions sharing a lower-cased surface form. Needs no annotation —
              the cheapest stand-in for predicted coreference.

The first two are ORACLE chains (gold annotation). Anything built on them — the prompt hint,
post-hoc propagation — is an upper bound on what a system with predicted chains would get.
"""
from __future__ import annotations

import glob
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

Chains = Dict[str, Tuple[str, ...]]

_ACTION_PREFIXES = ("ACTION", "NEG_ACTION")
_COREF_TAGS = ("CROSS_DOC_COREF", "INTRA_DOC_COREF")
MAVEN_SPAN_REPO = "Nofing/Maven-ERE-span"


def _chain_map(clusters: Iterable[Iterable[str]], present: Iterable[str]) -> Chains:
    present = set(present)
    out: Chains = {}
    for c in clusters:
        members = tuple(sorted(m for m in c if m in present))
        if len(members) >= 2:
            for m in members:
                out[m] = members
    return out


# ── ESC / ECB+ ────────────────────────────────────────────────────────────────

def _action_spans(root) -> Dict[str, tuple]:
    out = {}
    for el in root.find("Markables"):
        if el.tag.startswith(_ACTION_PREFIXES):
            span = tuple(t.get("t_id") for t in el.findall("token_anchor"))
            if span:
                out[el.get("m_id")] = span
    return out


def esc_clusters(esl_root: Path) -> Dict[str, List[List[str]]]:
    """{doc_id: [[esc_m_id, ...], ...]} — within-document event clusters of size >= 2,
    each cluster sorted numerically and clusters ordered by their first id."""
    esl_root = Path(esl_root)
    pattern = str(esl_root / "annotated_data" / "v0.9" / "*" / "*.xml.xml")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no ESC annotated_data under {esl_root} "
                                f"(clone github.com/tommasoc80/EventStoryLine there)")
    index: Dict[str, List[List[str]]] = {}
    for esc_path in paths:
        topic = Path(esc_path).parent.name
        doc_id = Path(esc_path).name.replace(".xml.xml", ".xml")
        ecb_path = esl_root / "ECB+_LREC2014" / "ECB+" / topic / doc_id
        if not ecb_path.exists():
            continue
        esc_by_span = {span: mid for mid, span in _action_spans(ET.parse(esc_path).getroot()).items()}
        ecb_root = ET.parse(ecb_path).getroot()
        ecb_spans = _action_spans(ecb_root)
        by_target: Dict[str, set] = {}
        relations = ecb_root.find("Relations")
        for rel in (relations if relations is not None else []):
            if rel.tag not in _COREF_TAGS or rel.find("target") is None:
                continue
            for src in rel.findall("source"):
                span = ecb_spans.get(src.get("m_id"))
                esc_mid = esc_by_span.get(span) if span is not None else None
                if esc_mid is not None:
                    by_target.setdefault(rel.find("target").get("m_id"), set()).add(esc_mid)
        clusters = [sorted(c, key=int) for c in by_target.values() if len(c) >= 2]
        if clusters:
            index[doc_id] = sorted(clusters, key=lambda c: int(c[0]))
    return index


# ── MAVEN span dataset ────────────────────────────────────────────────────────

def maven_span_rows(splits: Iterable[str] = ("test", "dev", "train")) -> Dict[str, dict]:
    """{doc_id: row} over the span dataset's splits (documents are disjoint across splits)."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    rows: Dict[str, dict] = {}
    for split in splits:
        path = hf_hub_download(MAVEN_SPAN_REPO, f"data/{split}-00000-of-00001.parquet",
                               repo_type="dataset")
        for r in pq.read_table(path).to_pylist():
            rows.setdefault(r["id"], r)
    return rows


def maven_clusters(span_row: dict, doc) -> List[List[str]]:
    """Coreference clusters of `doc` (eci mention ids) from its span-dataset row."""
    span_by_key = {tuple(x): m for m, x in zip(span_row["mentions"], span_row["spans"])}
    to_eci = {}
    for m, x in zip(doc.mentions, doc.spans):
        j = span_by_key.get(tuple(x))
        if j is not None:
            to_eci[j] = m
    rel = span_row["relations"]
    if isinstance(rel, str):
        rel = json.loads(rel)
    members = list(span_row["mentions"])
    parent: Dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in rel.get("coreference", []):
        ra, rb = find(members[a]), find(members[b])
        if ra != rb:
            parent[ra] = rb
    groups: Dict[str, List[str]] = defaultdict(list)
    for m in members:
        if m in parent:
            groups[find(m)].append(m)
    clusters = []
    for g in groups.values():
        mapped = sorted({to_eci[m] for m in g if m in to_eci})
        if len(mapped) >= 2:
            clusters.append(mapped)
    return clusters


# ── Surface form ──────────────────────────────────────────────────────────────

def surface_clusters(doc) -> List[List[str]]:
    groups: Dict[str, List[str]] = defaultdict(list)
    for mid, text in doc.surfaces.items():
        key = text.lower().strip()
        if key:
            groups[key].append(mid)
    return [sorted(g) for g in groups.values() if len(g) >= 2]


# ── Public API ────────────────────────────────────────────────────────────────

class ChainSource:
    """Lazily loads one chain source and serves clusters / chain maps per document."""

    def __init__(self, source: str, esl_root: Path | str | None = None):
        self.source = source
        self.esl_root = Path(esl_root) if esl_root else None
        self._esc: Dict[str, List[List[str]]] | None = None
        self._maven: Dict[str, dict] | None = None

    def clusters(self, doc) -> List[List[str]]:
        if self.source == "esc_ecb":
            if self._esc is None:
                self._esc = esc_clusters(self.esl_root)
            return self._esc.get(doc.id, [])
        if self.source == "maven_span":
            if self._maven is None:
                self._maven = maven_span_rows()
            row = self._maven.get(doc.id)
            return maven_clusters(row, doc) if row is not None else []
        if self.source == "surface":
            return surface_clusters(doc)
        raise ValueError(f"unknown chain source {self.source!r}")

    def chains(self, doc) -> Chains:
        return _chain_map(self.clusters(doc), set(doc.mentions) | set(doc.sentence_of))


# ── Prompt hint (formerly scripts/experiments/coref_run.py) ───────────────────

# Deliberately neutral: no "never link mentions of the same event" rule. ECB+ chains are
# coarser than ESC's event granularity, and 376 directed ESC gold positives sit inside a chain.
_HINT_HEADER = ("## Event coreference\n"
                "Mentions on the same line refer to the same real-world event, described more than once.")


def coref_block(doc, clusters: List[List[str]]) -> str:
    """Clusters rendered in the `<ID surface>` form the prompt uses for mentions.
    Byte-identical to coref_run.coref_block for the ESC runs made with it."""
    order = {m: i for i, m in enumerate(doc.mentions)}
    lines = []
    for c in clusters:
        present = sorted((m for m in c if m in doc.surfaces), key=lambda m: order.get(m, 10 ** 9))
        if len(present) >= 2:
            lines.append("  " + " ".join(f"<{m} {doc.surfaces[m]}>" for m in present))
    if not lines:
        return ""
    return _HINT_HEADER + "\n" + "\n".join(f"  E{i}:{l[1:]}" for i, l in enumerate(lines, 1))
