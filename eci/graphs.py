"""Causal graphs (TP/FP/FN per document) and the per-run histogram set.

Both read only the report: rows carry word and sentence distance and `doc_index` carries the
mention counts, so neither needs to re-download the dataset as agentere's did. Colours are
Okabe-Ito in both, so a TP is the same green in a graph and in a histogram.
"""
from __future__ import annotations

import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from eci.labels import CAUSED_BY, is_norel

TP_COLOR, FP_COLOR, FN_COLOR = "#009E73", "#E69F00", "#0072B2"
COLORS = {"TP": TP_COLOR, "FP": FP_COLOR, "FN": FN_COLOR}
STYLES = {"TP": "solid", "FP": "dotted", "FN": "dashed"}


# ── Causal graph ──────────────────────────────────────────────────────────────

def edges(gold, pred) -> List[dict]:
    """TP/FP/FN edges over (src, label, tgt) triples, drawn cause -> effect."""
    g, p = set(map(tuple, gold)), set(map(tuple, pred))

    def edge(t, status):
        src, label, tgt = t
        cause, effect = (tgt, src) if str(label).lower() == CAUSED_BY else (src, tgt)
        return {"src": cause, "tgt": effect, "status": status, "label": label}
    return ([edge(t, "TP") for t in sorted(g & p)] + [edge(t, "FP") for t in sorted(p - g)]
            + [edge(t, "FN") for t in sorted(g - p)])


def causal_graph(doc_id: str, surfaces: Dict[str, str], edge_list: List[dict],
                 doc_text: Optional[str] = None, fmt: str = "svg"):
    import graphviz
    g = graphviz.Digraph("causal_graph", format=fmt)
    g.attr(rankdir="LR", label=doc_id, labelloc="t", fontsize="16", pad="0.4,0.2",
           ranksep="0.4", nodesep="0.3", concentrate="true")
    g.attr("node", shape="box", style="rounded,filled", fillcolor="#fffde7", fontname="Helvetica", fontsize="11")
    g.attr("edge", fontname="Helvetica", fontsize="10", minlen="1")
    if doc_text:
        with g.subgraph(name="doc_text") as dt:
            dt.attr(rank="source")
            dt.node("doc_text", shape="note", style="filled", fillcolor="white", fontsize="10",
                    label=textwrap.fill(doc_text, width=100))
    used = {m for e in edge_list for m in (e["src"], e["tgt"])} or set(surfaces)
    for mid in sorted(used, key=lambda m: (len(m), m)):
        g.node(mid, label=f"{mid}\n{surfaces.get(mid, mid)}")
    for e in edge_list:
        g.edge(e["src"], e["tgt"], color=COLORS[e["status"]], style=STYLES[e["status"]],
               penwidth="2", label=e["label"])
    with g.subgraph(name="legend") as lg:
        lg.attr(rank="sink")
        lg.attr("node", shape="point", style="invis")
        lg.attr("edge", fontname="Helvetica", fontsize="10", penwidth="2")
        lg.node("legend_label", shape="plaintext", style="", label="Legend:")
        for status, text in (("TP", "correct causal"), ("FP", "hallucinated causal"), ("FN", "missed causal")):
            lg.node(f"legend_{status}_a")
            lg.node(f"legend_{status}_b")
            lg.edge(f"legend_{status}_a", f"legend_{status}_b", color=COLORS[status],
                    style=STYLES[status], label=f"{status} ({text})", fontcolor=COLORS[status])
    return g


def render_graph(out_dir: Path, doc_id: str, surfaces, gold, pred, doc_text=None) -> Optional[Path]:
    """Render to <out_dir>/<safe id>.svg. Never raises: a rendering failure must not cost the
    document its result."""
    import re
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^\w.-]", "_", doc_id)
        g = causal_graph(doc_id, surfaces, edges(gold, pred), doc_text=doc_text)
        return Path(g.render(str(out_dir / safe), cleanup=True))
    except Exception as e:
        print(f"[graphs] could not render the causal graph of {doc_id}: {e}")
        return None


# ── Histograms ────────────────────────────────────────────────────────────────

def _confusion(gold: str, pred: str) -> str:
    g, p = not is_norel(gold), not is_norel(pred)
    return "TP" if g and p else "FN" if g else "FP" if p else "TN"


def _grouped(ax, values: Dict[str, List[float]], weights: Optional[Dict[str, List[float]]],
             integer_valued: bool, title: str) -> float:
    import numpy as np
    allv = [v for vs in values.values() for v in vs]
    max_val = float(max(allv)) if allv else 0.0
    if integer_valued:
        n_bins = int(max_val) + 1 if max_val <= 100 else 60
        edges_ = np.linspace(0, max_val + (1 if max_val <= 100 else 0), n_bins + 1)
    else:
        edges_ = np.linspace(0, max_val if max_val > 0 else 1, 31)
    width = edges_[1] - edges_[0]
    centers = (edges_[:-1] + edges_[1:]) / 2
    bar = width / 4
    for i, c in enumerate(("TP", "FP", "FN")):
        counts, _ = np.histogram(values.get(c, []), bins=edges_,
                                 weights=(weights or {}).get(c) if weights else None)
        ax.bar(centers + (i - 1) * bar, counts, width=bar, label=c, color=COLORS[c])
    ax.set_title(title)
    ax.set_ylabel("relation count" if weights else "count")
    ax.legend()
    return max_val


def histograms(report: dict, label: str, out_dir: Path, prefix: str) -> List[Path]:
    """agentere's seven-figure set, over all pairs and again `intra.`-prefixed over
    same-sentence pairs only (whose pair-density denominator is the same-sentence pair count):
    word distance, sentence distance, relations per mention per document, relations per
    ordered pair per document, both again bucketed by GOLD density, and relations by mention
    count."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx = report["doc_index"]
    n_mentions = {d: len(v["mention_sentence"]) for d, v in idx.items()}
    n_intra = {d: sum(c * (c - 1) for c in Counter(v["mention_sentence"].values()).values())
               for d, v in idx.items()}
    rows = [r for r in report["per_pair_predictions"]
            if r["sentence_distance"] is not None and r["word_distance"] is not None]
    out: List[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        path = out_dir / f"{prefix}{name}"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        out.append(path)

    for intra_only in (False, True):
        sel = [r for r in rows if _confusion(r["gold"], r["pred"]) != "TN"
               and (not intra_only or r["sentence_distance"] == 0)]
        if not sel:
            continue
        pre = "intra." if intra_only else ""
        lab = f"{label} (intra-sentence only)" if intra_only else label

        kinds = ["intra"] if intra_only else ["intra", "inter"]
        fig, axes = plt.subplots(1, len(kinds), figsize=(6 * len(kinds), 4.5), squeeze=False)
        for ax, kind in zip(axes[0], kinds):
            vals = defaultdict(list)
            for r in sel:
                if (r["sentence_distance"] == 0) == (kind == "intra"):
                    vals[_confusion(r["gold"], r["pred"])].append(r["word_distance"])
            m = _grouped(ax, vals, None, True, f"{kind}-sentence pairs")
            ax.set_title(f"{kind}-sentence pairs (max={int(m)})")
            ax.set_xlabel("Word distance between mentions")
        fig.suptitle(f"Word distance between mentions — {lab}")
        save(fig, f"{pre}word_distance_histogram.png")

        fig, ax = plt.subplots(figsize=(6, 4.5))
        vals = defaultdict(list)
        for r in sel:
            vals[_confusion(r["gold"], r["pred"])].append(r["sentence_distance"])
        _grouped(ax, vals, None, True, "all pairs (intra only)" if intra_only else "all pairs (intra + inter)")
        ax.set_xlabel("Sentence distance between mentions")
        fig.suptitle(f"Sentence distance between mentions — {lab}")
        save(fig, f"{pre}sentence_distance_histogram.png")

        per_doc = defaultdict(Counter)
        for r in sel:
            per_doc[r["id"]][_confusion(r["gold"], r["pred"])] += 1
        docs = [d for d in per_doc if n_mentions.get(d)]

        def density_fig(denom, xlabel, name, title, by_gold):
            vals, wts = defaultdict(list), defaultdict(list)
            for d in docs:
                den = denom(d)
                if not den:
                    continue
                c = per_doc[d]
                for k in ("TP", "FP", "FN"):
                    x = (c["TP"] + c["FN"]) / den if by_gold else c[k] / den
                    vals[k].append(x)
                    wts[k].append(c[k])
            fig, ax = plt.subplots(figsize=(7, 4.5))
            _grouped(ax, vals, wts, False, title)
            ax.set_xlabel(xlabel)
            fig.suptitle(f"{title} — {lab}")
            save(fig, f"{pre}{name}")

        pair_denom = (lambda d: n_intra[d]) if intra_only else (lambda d: n_mentions[d] * (n_mentions[d] - 1))
        pair_label = ("positive relations / num. ordered same-sentence mention pairs" if intra_only
                      else "positive relations / (num. mentions * (num. mentions - 1))")
        density_fig(lambda d: n_mentions[d], "positive relations / num. mentions in document",
                    "density_per_document_histogram.png", "Relations per mention, per document", False)
        density_fig(pair_denom, pair_label, "density_per_pair_histogram.png",
                    "Relations per ordered pair, per document", False)
        density_fig(lambda d: n_mentions[d], "gold positive relations / num. mentions in document",
                    "density_per_document_by_gold_density_histogram.png",
                    "TP/FP/FN by gold relation density per mention", True)
        density_fig(pair_denom, f"gold {pair_label}", "density_per_pair_by_gold_density_histogram.png",
                    "TP/FP/FN by gold relation density per ordered pair", True)

        vals, wts = defaultdict(list), defaultdict(list)
        for d in docs:
            for k in ("TP", "FP", "FN"):
                vals[k].append(n_mentions[d])
                wts[k].append(per_doc[d][k])
        fig, ax = plt.subplots(figsize=(7, 4.5))
        _grouped(ax, vals, wts, True, "Relations by document mention count")
        ax.set_xlabel("number of mentions in document")
        fig.suptitle(f"Relations by number of mentions in document — {lab}")
        save(fig, f"{pre}num_mentions_per_document_histogram.png")
    return out
