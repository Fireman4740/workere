"""Training-free, strictly projected virtual event mention alias views.

All public model inputs are constructed by explicit allowlists. Never supply
corpus gold labels, relation annotations or reference clusters to this module.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from math import ceil
import json
import re
from typing import Any, Iterable


Pair = tuple[str, str]
ARMS = ("B", "R", "A", "P")
POSITIVE_LABELS = frozenset(("causes", "causedby"))
FORBIDDEN_INPUT_FIELDS = frozenset((
    "gold", "gold_relations", "gold_clusters", "gold_coreference", "annots",
    "relations", "labels", "positive_pairs", "coreference_gold", "annotation",
    "coref_clusters", "coref_instance", "causal_origin", "causal_frame", "plot_links",
))
COMMON_AUX_INSTRUCTION = (
    "The auxiliary records are a pointer index, NOT additional events. "
    "A handle_id points to exactly the original mention source_id at the "
    "specified token coordinates. Repeated or distinct handles for a source "
    "do not supply independent events or evidence. They do not imply "
    "coreference among different original mentions. Keep the source "
    "document and original candidate set authoritative. Output only "
    "text-supported causal relations; an empty list is valid."
)


def compact(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def digest(s: str) -> str:
    return sha256(s.encode("utf-8")).hexdigest()


def unordered(a: str, b: str) -> Pair:
    return (a, b) if a < b else (b, a)


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    tokens: tuple[str, ...]
    mentions: tuple[str, ...]
    spans: dict[str, tuple[int, ...]]
    candidates: frozenset[Pair]
    topic: str | None = None

    @classmethod
    def from_public_json(cls, raw: dict[str, Any]) -> "Document":
        if not isinstance(raw, dict):
            raise ValueError("Document must be an object")
        bad = FORBIDDEN_INPUT_FIELDS.intersection(raw)
        if bad:
            raise ValueError("Labels or gold in public input: " + ",".join(sorted(bad)))
        required = ("doc_id", "text", "tokens", "mentions", "spans", "candidates")
        absent = [k for k in required if k not in raw]
        if absent:
            raise ValueError("Missing required fields: " + ",".join(absent))
        doc_id = raw["doc_id"]
        if not isinstance(doc_id, str) or not doc_id:
            raise ValueError("doc_id must be a nonempty string")
        text = raw["text"]
        if not isinstance(text, str):
            raise ValueError("text must be a string")
        tok = raw["tokens"]
        if not isinstance(tok, list) or any(not isinstance(v, str) for v in tok):
            raise ValueError("tokens must be a list of strings")
        mids = raw["mentions"]
        if not isinstance(mids, list) or not mids or any(not isinstance(v, str) or not v for v in mids):
            raise ValueError("mentions must be a nonempty list of IDs")
        if len(mids) != len(set(mids)):
            raise ValueError("duplicate mention ID")
        if not isinstance(raw["spans"], dict) or set(raw["spans"]) != set(mids):
            raise ValueError("spans must map exactly every mention ID")
        spans: dict[str, tuple[int, ...]] = {}
        for mid in mids:
            seq = raw["spans"][mid]
            if not isinstance(seq, list) or not seq or any(type(i) is not int for i in seq):
                raise ValueError("spans must be nonempty token-index lists")
            if any(i < 0 or i >= len(tok) for i in seq):
                raise ValueError("span token index outside tokens")
            if seq != sorted(set(seq)):
                raise ValueError("span indices must be unique and sorted")
            spans[mid] = tuple(seq)
        cands = raw["candidates"]
        if not isinstance(cands, list):
            raise ValueError("candidates must be explicitly provided as list")
        cset: set[Pair] = set()
        for item in cands:
            if not isinstance(item, list) or len(item) != 2 or any(not isinstance(x, str) for x in item):
                raise ValueError("candidate must be [mention_id1, mention_id2]")
            a, b = item
            if a == b or a not in spans or b not in spans:
                raise ValueError("candidate self-loop or unknown ID")
            cset.add(unordered(a, b))
        topic = raw.get("topic")
        if topic is not None:
            topic = str(topic)
        return cls(doc_id, text, tuple(tok), tuple(mids), spans, frozenset(cset), topic)


def choose_mentions(d: Document, limit: int = 16) -> list[str]:
    k = min(limit, ceil(len(d.mentions) / 4))
    return sorted(d.mentions, key=lambda m: (digest("alias-v1|" + d.doc_id + "|" + m), m))[:k]


def make_p_renaming(d: Document) -> dict[str, str]:
    mids = sorted(d.mentions, key=lambda m: (digest("perm-v1|" + d.doc_id + "|" + m), m))
    mapped = {}
    taken = set(d.mentions)
    for i, original in enumerate(mids):
        candidate = "p__%04d" % i
        while candidate in taken:
            candidate += "_p"
        taken.add(candidate)
        mapped[original] = candidate
    return mapped


def remap_tags(text: str, mapping: dict[str, str]) -> str:
    """Remap only <mention_id token> annotation prefixes, never lexical text."""
    observed = set()

    def replace(m: re.Match[str]) -> str:
        old = m.group(1)
        if old in mapping:
            observed.add(old)
            return "<" + mapping[old]
        return m.group(0)

    result = re.sub(r"<([^\s<>]+)(?=[\s>])", replace, text)
    if observed != set(mapping):
        raise ValueError("P arm requires <ID token> markup for every mention; missing " +
                         ",".join(sorted(set(mapping) - observed)))
    return result


@dataclass
class View:
    doc_id: str
    arm: str
    document_text: str
    mentions: list[str]
    candidates: list[list[str]]
    auxiliary_records: list[dict[str, Any]]
    projection: dict[str, str]
    original_candidates: list[list[str]]
    original_source_sha256: str
    selected_source_mentions: list[str]
    common_instruction: str
    topic: str | None

    def to_json(self) -> dict[str, Any]:
        return self.__dict__.copy()


def prepare_view(d: Document, arm: str, selected: list[str] | None = None) -> View:
    if arm not in ARMS:
        raise ValueError("Unknown arm")
    selection = choose_mentions(d) if selected is None else selected
    if (len(selection) != len(set(selection)) or len(selection) > 16 or
            any(m not in d.spans for m in selection)):
        raise ValueError("Invalid selected mention")
    projection = {m: m for m in d.mentions}
    records: list[dict[str, Any]] = []
    source_to_display = {m: m for m in d.mentions}
    display = d.text
    if arm == "P":
        source_to_display = make_p_renaming(d)
        projection = {v: k for k, v in source_to_display.items()}
        display = remap_tags(d.text, source_to_display)
    elif arm in ("A", "R"):
        taken = set(d.mentions)
        for i, mid in enumerate(selection):
            if arm == "A":
                handle = "v__%04d" % i
                while handle in taken:
                    handle += "_v"
                taken.add(handle)
                projection[handle] = mid
            else:
                handle = mid
            records.append({
                "handle_id": handle,
                "source_id": mid,
                "offset_unit": "token_indices_zero_based",
                "source_token_indices": list(d.spans[mid]),
                "quoted_tokens": [d.tokens[j] for j in d.spans[mid]],
            })
    return View(
        doc_id=d.doc_id,
        arm=arm,
        document_text=display,
        mentions=[source_to_display[m] for m in d.mentions],
        candidates=[list(unordered(source_to_display[a], source_to_display[b]))
                    for a,b in sorted(d.candidates)],
        auxiliary_records=records,
        projection=projection,
        original_candidates=[list(pair) for pair in sorted(d.candidates)],
        original_source_sha256=digest(d.text),
        selected_source_mentions=list(selection) if arm in ("A", "R") else [],
        common_instruction=COMMON_AUX_INSTRUCTION if arm in ("A", "R") else "",
        topic=d.topic,
    )


def validate_pair_view(d: Document, a: View, r: View) -> dict[str, Any]:
    if a.arm != "A" or r.arm != "R" or a.doc_id != r.doc_id:
        raise ValueError("Need A and R views of same document")
    if a.selected_source_mentions != r.selected_source_mentions:
        raise ValueError("A/R must select exact same original mentions")
    if len(a.auxiliary_records) != len(r.auxiliary_records) or len(a.auxiliary_records) != len(a.selected_source_mentions):
        raise ValueError("A/R record cardinality differs")
    if a.original_candidates != r.original_candidates or a.original_candidates != [list(p) for p in sorted(d.candidates)]:
        raise ValueError("A/R candidate universes differ")
    for va, vr in zip(a.auxiliary_records, r.auxiliary_records):
        for k in ("source_id", "source_token_indices", "quoted_tokens", "offset_unit"):
            if va[k] != vr[k]:
                raise ValueError("A/R records differ beyond handles")
    if digest(a.document_text) != digest(d.text) or digest(r.document_text) != digest(d.text):
        raise ValueError("A/R changed the annotated document")
    return {
        "doc_id": d.doc_id,
        "source_hash": digest(d.text),
        "selected": len(a.selected_source_mentions),
        "a_record_chars": len(compact(a.auxiliary_records)),
        "r_record_chars": len(compact(r.auxiliary_records)),
        "requires_model_tokenizer_check": True,
    }


def project_one_generation(response: Any, v: View) -> tuple[set[Pair], Counter]:
    """Positive list only, outputs set and detailed rejections.

    Unknown IDs, nonpositive labels, self-loops and non-candidates are rejected.
    Duplicate alias edges cannot increase the per-pass vote.
    """
    if isinstance(response, dict):
        response = response.get("relations", None)
    if not isinstance(response, list):
        raise ValueError("Response must be a positive-triple list (or relations key)")
    allowed = {tuple(x) for x in v.original_candidates}
    preds: set[Pair] = set()
    counts: Counter = Counter()
    for triple in response:
        if not isinstance(triple, (tuple, list)) or len(triple) != 3:
            counts["bad_format"] += 1
            continue
        left, label, right = triple
        if not all(isinstance(x, str) for x in triple):
            counts["bad_format"] += 1
            continue
        if label.lower() not in POSITIVE_LABELS:
            counts["bad_label"] += 1
            continue
        if left not in v.projection or right not in v.projection:
            counts["unknown_id"] += 1
            continue
        a, b = v.projection[left], v.projection[right]
        if a == b:
            counts["self_loop"] += 1
            continue
        pair = unordered(a,b)
        if pair not in allowed:
            counts["not_candidate"] += 1
            continue
        if pair in preds:
            counts["duplicate_after_projection"] += 1
        preds.add(pair)
    counts["accepted_unique"] = len(preds)
    return preds, counts


def aggregate(responses: list[Any], v: View, slots: int = 7,
              threshold_votes: int = 4) -> dict[str, Any]:
    if len(responses) != slots:
        raise ValueError("Expected exactly %d response slots" % slots)
    if not 0 < threshold_votes <= slots:
        raise ValueError("invalid vote threshold")
    votes = Counter()
    rejects: Counter = Counter()
    for response in responses:
        if response is None:
            rejects["failed_or_empty_slot"] += 1
            continue
        try:
            pairs, counters = project_one_generation(response, v)
            votes.update(pairs)
            rejects.update(counters)
        except ValueError:
            rejects["invalid_response_slot"] += 1
    selected = sorted(pair for pair, count in votes.items() if count >= threshold_votes)
    return {
        "doc_id": v.doc_id,
        "arm": v.arm,
        "predicted_pairs": [list(p) for p in selected],
        "vote_counts": [{"pair":list(p), "count":votes[p], "share": votes[p]/slots}
                        for p in sorted(votes)],
        "counts": dict(rejects),
        "slots": slots,
        "threshold_votes": threshold_votes,
    }
