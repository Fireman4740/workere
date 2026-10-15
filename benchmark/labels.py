"""Canonical relation labels. Nothing else hard-codes these strings."""
from __future__ import annotations

CAUSES = "causes"
CAUSED_BY = "causedby"
NOREL = "norel"

# Produced by the binary collapse (causes + causedby -> one positive class).
BINARY_POS = "causal"

# Every string that means "no relation", compared after .lower(). Covers model output that
# deviates from the canonical "norel" ("None", "unknown") so it still counts as negative.
NOREL_VARIANTS = frozenset({"norel", "none", "no_rel", "unknown"})

CAUSAL = frozenset({CAUSES, CAUSED_BY})

DIRECTED_LABELS = [CAUSES, CAUSED_BY, NOREL]
BINARY_LABELS = [BINARY_POS, NOREL]


def is_norel(label: str) -> bool:
    return str(label).lower() in NOREL_VARIANTS


def normalize(label: str) -> str:
    """Case and whitespace normalisation of a model-emitted label.

    agentere counted "Causes", " causes" and "caused by" as labels distinct from "causes",
    so they split a pair's votes. Measured over all 70 agentere run logs: 0.03% of votes
    were off-spec, and normalising case/whitespace would change the plurality winner of 73
    pairs. Genuine misspellings ("cuses") are left alone and still count as positive in
    every binary metric, as they did before.
    """
    return "".join(str(label).split()).lower()
