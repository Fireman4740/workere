"""HuggingFace ECI datasets (`Nofing/*-standard-eci`) loaded into `Doc` objects.

Every repo shares one schema: `text` with inline `<ID surface>` mention markup, `annots` with
relation triples `<SRC_ID text> LABEL <TGT_ID text>`, and the token-level fields `tokens`,
`spans`, `mentions`, `sentences` (plus `pair_list` where the corpus defines candidate pairs).

Loading is byte-compatible with agentere's `dataprep.load_hf_dataset_parsed`: same gold
triples, same mention surfaces, same sentence index, same seeded document and pair shuffles —
so a workere run and an agentere run with the same config see the same documents, in the same
order, with the same prompts.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

from benchmark.labels import BINARY_POS, NOREL, is_norel

Triple = Tuple[str, str, str]

RELATION_TRIPLE_RE = re.compile(
    r"<([^>\s]+)\s+([^>\n]+)>"       # source tag: <ID content>
    r"\s+"
    r"([A-Za-z0-9_\-]+)"             # relation label
    r"\s+"
    r"<([^>\s]+)\s+([^>\n]+)>"       # target tag: <ID content>
)


@dataclass
class Doc:
    id: str
    idx: int                         # position in the split — k-fold and the `idx_` id fallback
    text: str                        # <ID surface>-marked text, never modified after load
    gold: List[Triple]
    lang: str
    mentions: List[str]              # mention ids, parallel to `spans`
    surfaces: Dict[str, str]         # mention id -> surface ("mentions_map" in agentere)
    sentence_of: Dict[str, int]      # mention id -> sentence index ("mention_sentence")
    first_token: Dict[str, int]      # mention id -> first token index
    pair_list: List[Tuple[str, str]] # candidate ordered pairs (dataset column or all i != j)
    tokens: List[str]
    spans: List[List[int]]
    sentences: List[List[int]]
    # Blocks appended after `text` in every prompt that shows this document (see prompt_text).
    syntax_block: str = ""
    coref_block: str = ""
    fold: int = field(default=-1)

    @property
    def prompt_text(self) -> str:
        """The document as the model reads it: text, then the syntax block, then the coref block.

        The single splice point for both blocks, used for the prediction prompt, the few-shot
        demonstrations, the CoT-synthesis input and the CoT dumps alike, so the same annotation
        is always present everywhere the document is shown.
        """
        out = self.text
        if self.syntax_block:
            out = f"{out}\n\n{self.syntax_block}"
        if self.coref_block:
            out = f"{out}\n\n{self.coref_block}"
        return out

    def relation_of(self, a: str, b: str) -> str:
        """'intra' | 'inter' | 'unknown' for the pair (a, b)."""
        sa, sb = self.sentence_of.get(a), self.sentence_of.get(b)
        if sa is None or sb is None:
            return "unknown"
        return "intra" if sa == sb else "inter"


def parse_annotations(ann_text: str) -> List[Triple]:
    if not ann_text:
        return []
    return [(m.group(1), m.group(3), m.group(4)) for m in RELATION_TRIPLE_RE.finditer(ann_text)]


def to_binary_undirected(triples: Iterable[Triple]) -> List[Triple]:
    """Collapse causes/causedby to BINARY_POS and keep one triple per unordered pair."""
    seen: Set[frozenset] = set()
    out: List[Triple] = []
    for src, lbl, tgt in triples:
        key = frozenset((src, tgt))
        if key in seen:
            continue
        seen.add(key)
        a, b = sorted((src, tgt))
        out.append((a, NOREL if is_norel(lbl) else BINARY_POS, b))
    return out


def doc_from_row(row: dict, i: int, *, text_field: str, ann_field: str,
                 valid_labels: Optional[Set[str]], binary_undirected: bool,
                 shuffle_pair_list: bool) -> Doc:
    row_id = str(row.get("id", f"idx_{i}"))
    tokens = list(row.get("tokens") or [])
    spans = [list(s) for s in (row.get("spans") or [])]
    mentions = list(row.get("mentions") or [])
    sentences = [list(s) for s in row["sentences"]]

    gold = parse_annotations(row.get(ann_field, ""))
    if binary_undirected:
        gold = to_binary_undirected(gold)
    elif valid_labels:
        gold = [(s, l, t) for s, l, t in gold if l in valid_labels]

    surfaces: Dict[str, str] = {}
    sentence_of: Dict[str, int] = {}
    first_token: Dict[str, int] = {}
    if tokens and spans and mentions and len(spans) == len(mentions):
        for mid, span in zip(mentions, spans):
            parts = [tokens[t] for t in span if 0 <= t < len(tokens)]
            if parts:
                surfaces[mid] = " ".join(parts)
            if span:
                first_token[mid] = span[0]
                for s_idx, (start, end) in enumerate(sentences):
                    if start <= span[0] < end:
                        sentence_of[mid] = s_idx
                        break

    pair_list: List[Tuple[str, str]] = []
    if mentions:
        raw_pairs = row.get("pair_list")
        if raw_pairs is not None:
            for pair in raw_pairs:
                if len(pair) == 2:
                    a, b = int(pair[0]), int(pair[1])
                    if a < len(mentions) and b < len(mentions):
                        pair_list.append((mentions[a], mentions[b]))
        else:
            pair_list = [(mentions[a], mentions[b])
                         for a in range(len(mentions)) for b in range(len(mentions)) if a != b]
    if shuffle_pair_list:
        random.Random(row_id).shuffle(pair_list)

    return Doc(id=row_id, idx=i, text=row.get(text_field, ""), gold=gold,
               lang=row.get("lang", "eng"), mentions=mentions, surfaces=surfaces,
               sentence_of=sentence_of, first_token=first_token, pair_list=pair_list,
               tokens=tokens, spans=spans, sentences=sentences)


def load_split(repo_id: str, split: str, *, text_field: str, ann_field: str,
               valid_labels: Optional[Set[str]], binary_undirected: bool,
               shuffle_pair_list: bool, revision: Optional[str] = None) -> List[Doc]:
    """Every document of a split, in split order."""
    from datasets import load_dataset
    ds = load_dataset(repo_id, split=split, **({"revision": revision} if revision else {}))
    return [doc_from_row(ds[i], i, text_field=text_field, ann_field=ann_field,
                         valid_labels=valid_labels, binary_undirected=binary_undirected,
                         shuffle_pair_list=shuffle_pair_list)
            for i in range(len(ds))]


def sample(docs: List[Doc], max_examples: int, shuffle: bool, seed: int) -> List[Doc]:
    """The evaluated documents: the first `max_examples` of the split, or of a seeded shuffle.

    The whole split is shuffled before the cut, so samples are nested: raising max_examples
    appends documents and never redraws the earlier ones. `Doc.idx` stays the position in the
    split, so folds and ids are unaffected by the shuffle.
    """
    order = list(range(len(docs)))
    if shuffle:
        random.Random(seed).shuffle(order)
    if max_examples > 0:
        order = order[:max_examples]
    return [docs[i] for i in order]


def label_set(cfg: dict) -> List[str]:
    from benchmark.labels import BINARY_LABELS
    return BINARY_LABELS if cfg["data"]["binary_undirected"] else list(cfg["data"]["labels"])


def load_eval_docs(cfg: dict) -> List[Doc]:
    ds = cfg["datasets"][cfg["active_dataset"]]
    docs = load_split(ds["repo_id"], ds["split"], text_field=ds["text_field"],
                      ann_field=ds["ann_field"], valid_labels=set(label_set(cfg)),
                      binary_undirected=cfg["data"]["binary_undirected"],
                      shuffle_pair_list=cfg["data"]["shuffle_pair_list"], revision=ds.get("revision"))
    return sample(docs, ds["max_examples"], cfg["data"]["shuffle_docs"], cfg["seed"])
