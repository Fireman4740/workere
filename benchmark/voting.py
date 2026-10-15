"""Resample passes -> one prediction per pair.

Plurality vote per directed pair. A pass that does not mention a pair casts an implicit
"norel" vote for it; a pass that lists a pair several times (models do loop and re-emit blocks
of their JSON on long documents) casts one vote, last label wins. Ties go to "norel", or to a
seeded random choice among the tied labels.
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple

from benchmark.labels import BINARY_POS, NOREL, is_norel

Triple = Tuple[str, str, str]


def aggregate(passes: Sequence[Sequence[Triple]], tie_breaking: str,
              rng: random.Random) -> Tuple[List[Triple], Dict[str, dict]]:
    """(winning triples, {"src,tgt": {"votes", "vote_counts", "final_label"}})."""
    pair_votes: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for run in passes:
        labels: Dict[Tuple[str, str], str] = {}
        for src, lab, tgt in run:
            labels[(src, tgt)] = lab
        for pair, lab in labels.items():
            pair_votes[pair].append(lab)

    n = len(passes)
    final: List[Triple] = []
    stats: Dict[str, dict] = {}
    for (src, tgt), votes in pair_votes.items():
        votes = votes + [NOREL] * (n - len(votes))
        counts = Counter(votes)
        top = max(counts.values())
        winners = [lbl for lbl, c in counts.items() if c == top]
        if len(winners) == 1:
            label = winners[0]
        else:
            label = NOREL if tie_breaking == NOREL else rng.choice(winners)
        if not is_norel(label):
            final.append((src, label, tgt))
        stats[f"{src},{tgt}"] = {"votes": votes, "vote_counts": dict(counts), "final_label": label}
    return final, stats


def single_pass_stats(triples: Sequence[Triple]) -> Dict[str, dict]:
    """The pair stats of a run without resampling: one vote for each emitted label."""
    return {f"{s},{t}": {"vote_counts": {lbl: 1}} for s, lbl, t in triples}


def votes_to_binary(pair_stats: Dict[str, dict], binary_default: str,
                    novote_norel: bool) -> List[Triple]:
    """Directed vote tallies -> one causal/norel decision per unordered pair.

    causes + causedby votes summed over BOTH directions are weighed against norel votes —
    not an OR over directions. With `novote_norel`, a direction that was never voted on at all
    counts as one implicit norel vote, so a lone causal vote is a tie, not an outright win.
    Ties resolve to `binary_default`. Pairs resolving to norel are omitted.
    """
    by_pair: Dict[frozenset, Dict[str, dict]] = defaultdict(dict)
    for key, st in pair_stats.items():
        src, tgt = key.split(",", 1)
        by_pair[frozenset((src, tgt))][key] = st["vote_counts"]
    out: List[Triple] = []
    for pair, by_direction in by_pair.items():
        if len(pair) < 2:
            continue  # "a,a": a mention cannot relate to itself (agentere crashed the doc here)
        causal = norel = 0
        for vc in by_direction.values():
            causal += sum(c for lbl, c in vc.items() if not is_norel(lbl))
            norel += sum(c for lbl, c in vc.items() if is_norel(lbl))
        if novote_norel and len(by_direction) < 2:
            norel += 1
        if causal > norel:
            label = BINARY_POS
        elif norel > causal:
            label = NOREL
        else:
            label = BINARY_POS if binary_default == BINARY_POS else NOREL
        if label != NOREL:
            a, b = sorted(pair)
            out.append((a, label, b))
    return out
