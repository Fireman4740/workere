"""Scope addenda appended to the test prompt, and the post-hoc filters that go with them.

All three leave the document text untouched — every `<ID surface>` tag stays — and add one
instruction after the template's own closing reminder, so each is the single changed variable
against the baseline:

  mention split    three calls per pass: pairs within the first half of the mentions (document
                   order), across the halves, within the second half. The blocks partition the
                   pair space exactly, so each pair still gets n_runs votes; an out-of-block
                   prediction is dropped (it would give a pair votes from two calls).
  whitelist        only the named mentions are in scope. "oracle" = mentions in >= 1 gold causal
                   pair (reads gold), "file" = ids from a JSON {doc_id: [ids]}. Obedience is
                   measured whether or not it is enforced.
  gold count       the exact number of gold causal relations (distinct pairs).

The obedience / leakage counts are returned with each document's result and summed by the
runner. agentere kept them in module-level dicts that were never reset, so two whitelist runs
queued in one process would have summed each other's counts.
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

from eci.data import Doc, Triple
from eci.labels import CAUSAL

# ── Mention split ─────────────────────────────────────────────────────────────

WITHIN_1, CROSS, WITHIN_2 = "within_1", "cross", "within_2"
BLOCK_KINDS = (WITHIN_1, CROSS, WITHIN_2)

_SPLIT_HEADER = """
SCOPE RESTRICTION — READ BEFORE ANSWERING
The event mentions of this document have been divided into two groups:
  Group 1: {group1}
  Group 2: {group2}
"""

_SPLIT_BODY = {
    WITHIN_1: """
This request covers GROUP 1 ONLY.
Report ONLY pairs where BOTH mentions are in Group 1.
A pair involving any Group 2 mention is OUT OF SCOPE here and is handled by a separate
request. Leave it out of your JSON even when the causal link is obvious.
""",
    CROSS: """
This request covers the GROUP 1 <-> GROUP 2 pairs ONLY.
Report ONLY pairs with ONE mention in Group 1 and the OTHER mention in Group 2.
A pair whose two mentions fall in the SAME group is OUT OF SCOPE here and is handled by
a separate request. Leave it out of your JSON even when the causal link is obvious.
""",
    WITHIN_2: """
This request covers GROUP 2 ONLY.
Report ONLY pairs where BOTH mentions are in Group 2.
A pair involving any Group 1 mention is OUT OF SCOPE here and is handled by a separate
request. Leave it out of your JSON even when the causal link is obvious.
""",
}


def split_blocks(doc: Doc) -> List[dict]:
    """The three call descriptors of one document. Mentions without a surface are not rendered
    in the text, so they are left out; an odd count puts the extra mention in group 1."""
    ids = [m for m in doc.mentions if m in doc.surfaces]
    half = (len(ids) + 1) // 2
    g1, g2 = ids[:half], ids[half:]
    header = _SPLIT_HEADER.format(group1=", ".join(g1) or "(none)", group2=", ".join(g2) or "(none)")
    return [{"kind": k, "group1": set(g1), "group2": set(g2),
             "instruction": (header + _SPLIT_BODY[k]).strip()} for k in BLOCK_KINDS]


def in_block(src: str, tgt: str, block: dict) -> bool:
    g1, g2 = block["group1"], block["group2"]
    if block["kind"] == WITHIN_1:
        return src in g1 and tgt in g1
    if block["kind"] == WITHIN_2:
        return src in g2 and tgt in g2
    return (src in g1 and tgt in g2) or (src in g2 and tgt in g1)


def filter_block(triples: Sequence[Triple], block: dict) -> Tuple[List[Triple], int]:
    kept = [t for t in triples if in_block(t[0], t[2], block)]
    return kept, len(triples) - len(kept)


# ── Mention whitelist ─────────────────────────────────────────────────────────

_WHITELIST = """
SCOPE RESTRICTION — READ BEFORE ANSWERING
Only these event mentions are in scope for this document:
  {kept}

Reason over pairs drawn from that list ONLY. Every other `<ID text>` tag in the text above is
out of scope: do not output any pair that contains one, in either position. The text still
shows them because it is unmodified — treat them as ordinary words, not as candidates.
""".strip()


@lru_cache(maxsize=4)
def _whitelist_file(path: str) -> Dict[str, List[str]]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def whitelist_ids(doc: Doc, mode: str, path: str) -> List[str]:
    """In-scope mention ids, in document order."""
    mentions = doc.mentions or list(doc.surfaces)
    if mode == "oracle":
        involved = {m for s, l, t in doc.gold if l in CAUSAL for m in (s, t)}
        kept = [m for m in mentions if m in involved]
        missing = involved - set(kept)
        if missing:  # the oracle would be capping its own recall
            raise ValueError(f"{doc.id}: gold endpoints not in the mention list: {sorted(missing)}")
        return kept
    allowed = set(_whitelist_file(path).get(doc.id, []))
    return [m for m in mentions if m in allowed]


def whitelist_instruction(kept: Sequence[str], doc: Doc) -> Optional[str]:
    """None when the list restricts nothing: the block would then be an instruction the
    baseline never sees while changing no scope — exactly the confound to avoid."""
    mentions = doc.mentions or list(doc.surfaces)
    if not kept or len(kept) >= len(mentions):
        return None
    return _WHITELIST.format(kept=", ".join(f"<{m} {doc.surfaces[m]}>" if m in doc.surfaces else m
                                            for m in kept))


def enforce_whitelist(triples: Sequence[Triple], kept: Sequence[str]) -> Tuple[List[Triple], int]:
    allowed = set(kept)
    out = [t for t in triples if t[0] in allowed and t[2] in allowed]
    return out, len(triples) - len(out)


# ── Gold count ────────────────────────────────────────────────────────────────

def gold_count(doc: Doc) -> int:
    """Distinct causal pairs, each counted once whatever its direction."""
    return len({frozenset((s, t)) for s, l, t in doc.gold if l in CAUSAL})


def count_instruction(n: int) -> str:
    if n == 0:
        return ("## Number of causal relations\n"
                "This document contains no causal relations: the final JSON array is empty [].")
    s = "s" if n != 1 else ""
    return ("## Number of causal relations\n"
            f"This document contains exactly {n} causal relation{s}: "
            f"{n} distinct pair{s} of event mentions, intra- or inter-sentence. "
            f"Each is listed in both directions, so the final JSON array holds exactly {2 * n} entries.")
