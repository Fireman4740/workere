"""Vote-threshold sweep, optionally with coreference propagation and transitive closure.

At threshold t an ordered pair is predicted causal when >= t passes voted causes/causedby for
it. t = 1 is the union of the passes, t = n the unanimous vote; the shipped plurality vote sits
at n // 2 + 1 for odd n (it can differ from that point on three-way ties).

Scoring convention — the one thing to keep in mind when comparing numbers
------------------------------------------------------------------------
"ordered" (default) scores ordered pairs, each causal link counted in both directions as the
gold stores it — the convention of the logged `binary*` metrics and of agentere's
vote_threshold*.py, vote_threshold_coref.py and maven_posthoc_sweep.py. "undirected" scores
unordered pairs with max(votes of the two directions) — the convention of agentere's
posthoc_coref_report.py, whose numbers are therefore NOT comparable with the others'.

Variants (all post-hoc, zero model calls):
  base            the votes as they are
  +coref          each predicted ordered pair expanded over both mentions' coreference chains —
                  how the ESC and MAVEN gold were built from their annotator links
                  (= agentere vote_threshold_coref.py; --drop-in-chain skips same-event pairs)
  +coref_edges    the same on DIRECTED edges read off the vote labels, re-materialised in both
                  orientations (= agentere maven_posthoc_sweep.py "+coref")
  +trans / +trans1 / +trans1(unanim) / +coref+trans   directed transitive closure a->b, b->c
                  => a->c: to a fixpoint, one step, one step on unanimous premises only, and
                  after propagation (= maven_posthoc_sweep.py)
Propagated or derived pairs that no report row mentions are scored too — as false positives
when they are not gold — so a transform cannot inflate precision by construction.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple

from eci.metrics import prf
from eci.posthoc.runs import DocVotes, Pair, RunVotes

SCOPES = ("all", "intra", "inter")
CHAIN_VARIANTS = ("+coref", "+coref_edges", "+coref+trans")
ALL_VARIANTS = ("base", "+coref", "+coref_edges", "+trans", "+trans1", "+trans1(unanim)", "+coref+trans")


def predicted(doc: DocVotes, t: int) -> Set[Pair]:
    return {p for p, c in doc.votes.items() if sum(c.values()) >= t}


def edges_at(doc: DocVotes, t: int) -> Set[Pair]:
    """Directed edges x -> y from the pairs predicted at t: (a,b) "causes" and (b,a) "causedby"
    both assert a -> b. A pair whose two causal labels tie contributes both directions."""
    out = set()
    for (a, b), c in doc.votes.items():
        if sum(c.values()) < t:
            continue
        top = max(c.values())
        if c.get("causes", 0) == top:
            out.add((a, b))
        if c.get("causedby", 0) == top:
            out.add((b, a))
    return out


def materialise(edges: Iterable[Pair]) -> Set[Pair]:
    return {x for a, b in edges for x in ((a, b), (b, a))}


def propagate(pairs: Iterable[Pair], chains: dict, drop_in_chain: bool = False) -> Set[Pair]:
    out = set()
    for a, b in pairs:
        for x in chains.get(a, (a,)):
            for y in chains.get(b, (b,)):
                if x == y:
                    continue
                if drop_in_chain and x in chains and chains.get(x) == chains.get(y):
                    continue
                out.add((x, y))
    return out


def transitive(edges: Set[Pair], steps: int = 0, cap: int = 200_000) -> Set[Pair]:
    out = set(edges)
    succ = defaultdict(set)
    for a, b in out:
        succ[a].add(b)
    rounds = 0
    while True:
        rounds += 1
        new = {(a, c) for a, b in out for c in succ.get(b, ()) if c != a and (a, c) not in out}
        if not new or len(out) + len(new) > cap:
            break
        out |= new
        for a, b in new:
            succ[a].add(b)
        if steps and rounds >= steps:
            break
    return out


def variant_pairs(doc: DocVotes, t: int, n_runs: int, variant: str, chains: dict,
                  drop_in_chain: bool) -> Set[Pair]:
    base = predicted(doc, t)
    if variant == "base":
        return base
    if variant == "+coref":
        return propagate(base, chains, drop_in_chain)
    e = edges_at(doc, t)
    if variant == "+coref_edges":
        return base | materialise(propagate(e, chains))
    if variant == "+trans":
        return base | materialise(transitive(e))
    if variant == "+trans1":
        return base | materialise(transitive(e, steps=1))
    if variant == "+trans1(unanim)":
        hi = edges_at(doc, n_runs)
        return base | materialise({(a, c) for a, b in hi for b2, c in hi if b2 == b and c != a} | e)
    if variant == "+coref+trans":
        return base | materialise(transitive(propagate(e, chains)))
    raise ValueError(variant)


def score(docs: Iterable[DocVotes], pred_by_doc: Dict[str, Set[Pair]], undirected: bool) -> dict:
    """{scope: (P, R, F1, tp, fp, fn)} in percent."""
    agg = {s: [0, 0, 0] for s in SCOPES}
    for d in docs:
        gold = set(d.gold)
        pred = pred_by_doc.get(d.id, set())
        if undirected:
            gold = {tuple(sorted(p)) for p in gold}
            pred = {tuple(sorted(p)) for p in pred}
        for pair in gold | pred:
            g, p = pair in gold, pair in pred
            for s in ("all", d.rel(*pair)):
                if s == "unknown":
                    continue
                agg[s][0 if (g and p) else 1 if p else 2] += 1
    out = {}
    for s, (tp, fp, fn) in agg.items():
        P, R, F = prf(tp, fp, fn)
        out[s] = (100 * P, 100 * R, 100 * F, tp, fp, fn)
    return out


def undirected_votes(doc: DocVotes) -> DocVotes:
    """posthoc_coref_report's view: one unordered pair, max votes of the two directions."""
    votes: Dict[Pair, Counter] = {}
    for (a, b), c in doc.votes.items():
        key = tuple(sorted((a, b)))
        if key not in votes or sum(c.values()) > sum(votes[key].values()):
            votes[key] = c
    return DocVotes(doc.id, doc.gold, votes, doc.all_votes, doc.row_rel, doc.sentence_of,
                    doc.surfaces, doc.first_token)


def sweep(run: RunVotes, variants: List[str], chains_by_doc: Optional[Dict[str, dict]] = None,
          keep: Optional[Set[str]] = None, undirected: bool = False,
          drop_in_chain: bool = False) -> Dict[int, dict]:
    """{t: {variant: {scope: (P, R, F1, tp, fp, fn)}, "_emitted": {variant: n}}}."""
    docs = [d for d_id, d in run.docs.items() if keep is None or d_id in keep]
    if undirected:
        docs = [undirected_votes(d) for d in docs]
    chains_by_doc = chains_by_doc or {}
    out = {}
    for t in range(1, run.n_runs + 1):
        row, emitted = {}, {}
        for v in variants:
            preds = {d.id: variant_pairs(d, t, run.n_runs, v, chains_by_doc.get(d.id, {}), drop_in_chain)
                     for d in docs}
            row[v] = score(docs, preds, undirected)
            emitted[v] = sum(len(p) for p in preds.values())
        row["_emitted"] = emitted
        out[t] = row
    return out


def chains_for(run: RunVotes, source: str, esl_root) -> Dict[str, dict]:
    """{doc_id: {mention: chain}} for the run's documents."""
    from eci.chains import ChainSource
    src = ChainSource(source, esl_root)
    docs = run.eval_docs()
    return {doc_id: src.chains(doc) for doc_id, doc in docs.items()}


def default_chain_source(run: RunVotes) -> Optional[str]:
    repo = run.cfg["datasets"][run.dataset]["repo_id"].lower()
    if "eventstoryline" in repo:
        return "esc_ecb"
    if "maven" in repo:
        return "maven_span"
    return None


def format_table(run: RunVotes, result: Dict[int, dict], variants: List[str], label: str) -> str:
    n = run.n_runs
    shipped = n // 2 + 1
    lines = [f"=== {label}",
             f"{'t':>4} {'variant':16}{'ALL  P / R / F1':>22}{'INTRA  P / R / F1':>24}"
             f"{'INTER  P / R / F1':>24}{'pairs':>9}"]
    best: Dict[str, Tuple[int, float]] = {}
    for t, row in result.items():
        tag = {1: "union", n: "unanimous"}.get(t, "")
        if n > 1 and t == shipped:
            tag = "majority*"
        for v in variants:
            cells = "".join(f"  {row[v][s][0]:6.1f}/{row[v][s][1]:5.1f}/{row[v][s][2]:5.1f}" for s in SCOPES)
            lines.append(f">={t:<2} {v:16}{cells}{row['_emitted'][v]:9d}" + (f"   {tag}" if v == variants[0] and tag else ""))
            if v not in best or row[v]["all"][2] > best[v][1]:
                best[v] = (t, row[v]["all"][2])
        if len(variants) > 1:
            lines.append("")
    lines.append("best ALL F1: " + "  ".join(f"{v} {f:.1f} (t>={t})" for v, (t, f) in best.items()))
    return "\n".join(lines)


def sanity_line(run: RunVotes, result: Dict[int, dict]) -> Optional[str]:
    """The majority point next to the metric the run logged: if they disagree by more than a
    few tenths, the sweep is not measuring what the run reported."""
    if run.n_runs < 2 or "binary" not in run.results:
        return None
    P, R, F, *_ = result[run.n_runs // 2 + 1]["base"]["all"]
    b = run.results["binary"]
    return (f"sanity (majority vs logged binary): {P:.1f}/{R:.1f}/{F:.1f} vs "
            f"{100 * b['precision']:.1f}/{100 * b['recall']:.1f}/{100 * b['f1']:.1f}")
