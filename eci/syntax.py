"""spaCy-derived syntactic annotation, appended after the document text.

Two orthogonal settings (`syntax.level`, `syntax.discourse`) select what the block contains:

  level      "off" | "mentions" | "args" | "paths" — mention-centric and cumulative; "paths"
             is quadratic in mentions per sentence, which is what the size caps bound.
  discourse  any of "skeleton", "participants", "pos" — document-linear, and the only part of
             the block that describes the ~47% of sentences holding no event mention.

What the parser is fed: `" ".join(doc.tokens)` — the document without its `<ID surface>`
markup (stripping the markup from `text` reproduces the joined tokens byte-for-byte on every
document of all four repos). spaCy re-tokenises that string, so the dataset's `spans` and
`sentences` are translated to spaCy token indices through exact character offsets before
anything renders; sentence *membership* stays the dataset's, so `S<n>` labels agree with
`Doc.sentence_of` and `few_shot.intra_only`.

English only (`en_core_web_sm`). Other languages get no block rather than a nonsense parse; a
missing spaCy install degrades to no block everywhere. The block is stored on
`Doc.syntax_block` and reaches every prompt through `Doc.prompt_text`; `Doc.text` itself is
never touched. Output is byte-identical to agentere's utils/syntax.py.
"""
from __future__ import annotations

import sys
import threading
from collections import deque
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

HEADER = "### Syntax"
DISCOURSE_ORDER = ("skeleton", "participants", "pos")

_MODEL = "en_core_web_sm"
_ENGLISH = frozenset({"en", "eng", "english"})

# ClearNLP / OntoNotes labels, as emitted by en_core_web_sm (no `obj`, no `obl`).
_SUBJ = frozenset({"nsubj", "nsubjpass", "csubj", "csubjpass"})
_OBJ = frozenset({"dobj", "dative", "attr", "oprd", "ccomp", "xcomp"})
_OBL = frozenset({"prep", "agent", "npadvmod"})
_CONNECTIVE_DEPS = frozenset({"mark", "prep", "cc", "advmod"})
_PASSIVE_DEPS = frozenset({"nsubjpass", "auxpass"})
_MORPH_KEYS = ("Tense", "VerbForm")

# Size caps. Uncapped, one Causal-TimeBank document emitted 559 path lines around a 1910-token
# body, and a 275-mention document a 58,000-character block. Paths of length <= 5 are 87% of
# all paths on both English corpora; the long remainder is the weakest evidence and the most
# verbose to render.
_MAX_ARGS = 3
_MAX_FILLER_TOKENS = 10
_MAX_CONNECTIVES = 12
_MAX_PATH_LEN = 5
_MAX_PATHS_PER_SENTENCE = 30
_MAX_PATHS_PER_DOC = 100
_MAX_MENTION_LINES = 60
_MAX_BLOCK_CHARS = 8000
_MAX_PARTICIPANTS = 15

# spaCy's Language is safe for neither concurrent load nor concurrent use.
_LOCK = threading.Lock()
_warned: set = set()


def _warn_once(key: str, message: str) -> None:
    if key not in _warned:
        _warned.add(key)
        print(message, file=sys.stderr)


def _lang_key(lang: str) -> str:
    key = (lang or "eng").strip().lower()
    return key[len("causal-"):] if key.startswith("causal-") else key


@lru_cache(maxsize=2)
def _load_pipeline(with_ner: bool = False):
    try:
        import spacy
    except ImportError:
        _warn_once("no-spacy", "[syntax] spaCy is not installed — no annotation will be appended.")
        return None
    try:
        return spacy.load(_MODEL) if with_ner else spacy.load(_MODEL, exclude=["ner"])
    except OSError:
        _warn_once("no-model", f"[syntax] spaCy model {_MODEL!r} is not installed — "
                               f"no annotation will be appended.")
        return None


# ── Doc construction ──────────────────────────────────────────────────────────

def build_parse(nlp, tokens: Sequence[str]):
    """Parse the plain document `" ".join(tokens)`; return (sdoc, starts, ends), where
    starts[i]/ends[i] is the spaCy token range covering dataset token i. Token i starts at
    sum(len(t) + 1 for t in tokens[:i]) because the join is single-spaced."""
    text = " ".join(tokens)
    sdoc = nlp(text)
    starts: List[int] = []
    ends: List[int] = []
    off = 0
    for tok in tokens:
        span = sdoc.char_span(off, off + len(tok), alignment_mode="expand")
        if span is None:
            starts.append(len(sdoc))
            ends.append(len(sdoc))
        else:
            starts.append(span.start)
            ends.append(span.end)
        off += len(tok) + 1
    return sdoc, starts, ends


def _aligned_view(doc, sdoc, starts: Sequence[int], ends: Sequence[int]) -> dict:
    """The fields the renderer reads, with `spans` and `sentences` in spaCy token indices.

    `sentences` keeps its length and order: the block's S<n> labels are positions in it and
    must agree with Doc.sentence_of. A sentence spaCy covers with nothing becomes empty.
    """
    n = len(starts)
    spans = [sorted({j for i in span if 0 <= int(i) < n for j in range(starts[int(i)], ends[int(i)])})
             for span in doc.spans]
    sentences: List[Tuple[int, int]] = []
    for sent in doc.sentences:
        a, b = int(sent[0]), int(sent[1])
        if 0 <= a < n and b > a:
            sentences.append((starts[a], ends[min(b, n) - 1]))
        else:
            edge = starts[a] if 0 <= a < n else len(sdoc)
            sentences.append((edge, edge))
    return {"spans": spans, "sentences": sentences, "mentions": doc.mentions,
            "mentions_map": doc.surfaces, "mention_sentence": doc.sentence_of,
            "pair_list_ids": doc.pair_list}


def _mention_index(view: dict) -> Tuple[Dict[int, str], Dict[str, List[int]]]:
    tok2mention: Dict[int, str] = {}
    mention_tokens: Dict[str, List[int]] = {}
    for mid, span in zip(view["mentions"], view["spans"]):
        if not span or mid not in view["mentions_map"]:
            continue
        idxs = [int(i) for i in span]
        mention_tokens[mid] = idxs
        for i in idxs:
            tok2mention[i] = mid
    return tok2mention, mention_tokens


def _span_root(sdoc, idxs: Sequence[int]):
    """Head token of a (possibly discontinuous) mention span."""
    inside = set(idxs)
    for i in idxs:
        tok = sdoc[i]
        if tok.head.i not in inside or tok.head.i == i:
            return tok
    return sdoc[idxs[0]]


# ── Rendering helpers ─────────────────────────────────────────────────────────

def _mention_label(mid: str, mentions_map: Dict[str, str]) -> str:
    return f"<{mid} {mentions_map[mid]}>"


def _token_label(i: int, sdoc, tok2mention, mentions_map) -> str:
    mid = tok2mention.get(i)
    return _mention_label(mid, mentions_map) if mid else sdoc[i].text


def _render_range(lo: int, hi: int, sdoc, tok2mention, mentions_map,
                  exclude: frozenset = frozenset()) -> str:
    """Tokens [lo, hi] with mention markup re-applied (each mention once), truncated."""
    parts: List[str] = []
    emitted: set = set()
    truncated = False
    for i in range(lo, hi + 1):
        if i in exclude:
            continue
        mid = tok2mention.get(i)
        if mid:
            if mid in emitted:
                continue
            emitted.add(mid)
            parts.append(_mention_label(mid, mentions_map))
        else:
            parts.append(sdoc[i].text)
        if len(parts) >= _MAX_FILLER_TOKENS and i < hi:
            truncated = True
            break
    return " ".join(parts) + (" ..." if truncated else "")


def _morph(tok) -> str:
    """Tense/VerbForm plus Voice, Polarity and modality read off the children (the English
    model populates neither Voice nor Polarity in `morph`)."""
    vals = []
    for key in _MORPH_KEYS:
        got = tok.morph.get(key)
        if got:
            vals.append(f"{key}={got[0]}")
    if any(c.dep_ in _PASSIVE_DEPS for c in tok.children):
        vals.append("Voice=Pass")
    if any(c.dep_ == "neg" for c in tok.children):
        vals.append("Polarity=Neg")
    modal = next((c.text for c in tok.children if c.dep_ == "aux" and c.tag_ == "MD"), None)
    if modal:
        vals.append(f"Modal={modal}")
    return "|".join(vals)


_ROLE_ORDER = {"subj": 0, "obj": 1, "obl": 2}


def _args_of(root, own_tokens: frozenset, sdoc, tok2mention, mentions_map) -> str:
    found: List[Tuple[int, str, str]] = []
    for child in root.children:
        if child.i in own_tokens:
            continue
        if child.dep_ in _SUBJ:
            role = "subj"
        elif child.dep_ in _OBJ:
            role = "obj"
        elif child.dep_ in _OBL:
            role = "obl"
        else:
            continue
        idxs = [t.i for t in child.subtree]
        filler = _render_range(min(idxs), max(idxs), sdoc, tok2mention, mentions_map,
                               exclude=own_tokens)
        if filler:
            found.append((_ROLE_ORDER[role], role, filler))
    found.sort(key=lambda f: (f[0], f[2]))
    return " ".join(f'{role}="{filler}"' for _, role, filler in found[:_MAX_ARGS])


def _connectives(start: int, end: int, sdoc, tok2mention, mentions_map) -> str:
    out: List[str] = []
    for i in range(start, end):
        tok = sdoc[i]
        if tok.dep_ not in _CONNECTIVE_DEPS:
            continue
        head = _token_label(tok.head.i, sdoc, tok2mention, mentions_map)
        out.append(f"{tok.text}({tok.dep_} -> {head})")
        if len(out) >= _MAX_CONNECTIVES:
            break
    return " ".join(out)


def _dep_path(sdoc, a, b, start: int, end: int) -> Optional[List[Tuple[int, str, bool]]]:
    """Shortest undirected dependency path a -> b confined to [start, end)."""
    if a.i == b.i:
        return None
    prev: Dict[int, Tuple[int, str, bool]] = {a.i: (-1, "", False)}
    queue = deque([a.i])
    while queue:
        cur = queue.popleft()
        if cur == b.i:
            break
        tok = sdoc[cur]
        neighbours: List[Tuple[int, str, bool]] = []
        if tok.head.i != cur and start <= tok.head.i < end:
            neighbours.append((tok.head.i, tok.dep_, True))
        for child in tok.children:
            if start <= child.i < end:
                neighbours.append((child.i, child.dep_, False))
        for nxt, label, up in neighbours:
            if nxt not in prev:
                prev[nxt] = (cur, label, up)
                queue.append(nxt)
    if b.i not in prev:
        return None
    path: List[Tuple[int, str, bool]] = []
    node = b.i
    while node != a.i:
        parent, label, up = prev[node]
        path.append((node, label, up))
        node = parent
    path.reverse()
    return path


def _format_path(a_label: str, path, sdoc, tok2mention, mentions_map) -> str:
    parts = [a_label]
    for idx, label, up in path:
        parts.append(f"-{label}->" if up else f"<-{label}-")
        parts.append(_token_label(idx, sdoc, tok2mention, mentions_map))
    return " ".join(parts) + f" (len={len(path)})"


# ── Document-linear sections ──────────────────────────────────────────────────

def _sentence_root(sdoc, start: int, end: int):
    """Head of a sentence range, preferring a verbal root governing most of the range."""
    roots = [sdoc[i] for i in range(start, end)
             if sdoc[i].head.i == i or not (start <= sdoc[i].head.i < end)]
    if not roots:
        return sdoc[start]
    return max(roots, key=lambda t: (t.pos_ in ("VERB", "AUX"),
                                     sum(1 for c in t.subtree if start <= c.i < end)))


def _skeleton(view: dict, sdoc, mention_tokens: Dict[str, List[int]]) -> List[str]:
    counts: Dict[int, int] = {}
    for mid in mention_tokens:
        si = view["mention_sentence"].get(mid)
        if si is not None:
            counts[si] = counts.get(si, 0) + 1
    out: List[str] = []
    for si, (start, end) in enumerate(view["sentences"]):
        root = _sentence_root(sdoc, start, end)
        subj = next((c for c in root.children if c.dep_ in _SUBJ), None)
        obj = next((c for c in root.children if c.dep_ in _OBJ), None)
        n = counts.get(si, 0)
        if not n and root.pos_ not in ("VERB", "AUX") and subj is None and obj is None:
            continue
        bits = [f"  S{si + 1}", f"pred={root.lemma_}"]
        morph = _morph(root)
        if morph:
            bits.append(morph)
        bits.append(f'subj="{subj.text}"' if subj is not None else "subj=-")
        bits.append(f'obj="{obj.text}"' if obj is not None else "obj=-")
        bits.append(f"events={n}")
        out.append(" ".join(bits))
    return out


def _participants(view: dict, sdoc) -> List[str]:
    where: Dict[Tuple[str, str], set] = {}
    for ent in sdoc.ents:
        for si, (start, end) in enumerate(view["sentences"]):
            if start <= ent.start < end:
                where.setdefault((ent.text, ent.label_), set()).add(si + 1)
                break
    if not where:
        return []
    ranked = sorted(where.items(), key=lambda kv: (-len(kv[1]), min(kv[1]), kv[0][0]))
    out: List[str] = []
    for (text, label), sents in ranked[:_MAX_PARTICIPANTS]:
        marks = ", ".join(f"S{i}" for i in sorted(sents))
        out.append(f'  "{text}" ({label}) {marks}' + ("  <- recurs" if len(sents) > 1 else ""))
    omitted = len(ranked) - len(out)
    if omitted > 0:
        out.append(f"  ({omitted} further entity/entities omitted)")
    return out


def _pos_text(view: dict, sdoc) -> List[str]:
    out: List[str] = []
    for si, (start, end) in enumerate(view["sentences"]):
        pieces = [f"{sdoc[i].text}/{sdoc[i].tag_}" for i in range(start, end)]
        if pieces:
            out.append(f"  S{si + 1} " + " ".join(pieces))
    return out


def render(view: dict, sdoc, lvl: str, comps: Tuple[str, ...] = ()) -> str:
    mentions_map = view["mentions_map"]
    mention_sentence = view["mention_sentence"]
    tok2mention, mention_tokens = _mention_index(view)
    if not mention_tokens and not comps:
        return ""

    header = f"{HEADER} (spaCy {_MODEL}"
    if lvl != "off":
        header += "; dep = relation to the head token"
    if lvl == "paths":
        header += "; in paths, A -dep-> B means A is a dependent of B, A <-dep- B the reverse"
    header += ")"
    lines: List[str] = [header]
    mention_tokens_all = mention_tokens
    if lvl == "off":
        mention_tokens = {}

    # Document-linear sections first: the character cap truncates from the end, and these are
    # bounded and cheap while the "paths" list is the quadratic part.
    if "skeleton" in comps:
        rows = _skeleton(view, sdoc, mention_tokens_all)
        if rows:
            lines.append("### Discourse (one row per sentence; events = event mentions in it)")
            lines.extend(rows)
    if "participants" in comps:
        rows = _participants(view, sdoc)
        if rows:
            lines.append("### Participants (named entities and the sentences they appear in)")
            lines.extend(rows)

    by_sentence: Dict[int, List[str]] = {}
    for mid in mention_tokens:
        by_sentence.setdefault(mention_sentence.get(mid, -1), []).append(mid)

    paths_budget = _MAX_PATHS_PER_DOC
    mention_budget = _MAX_MENTION_LINES
    omitted_mentions = 0
    pair_ids = view["pair_list_ids"] or []

    for sent_idx, (start, end) in enumerate(view["sentences"]):
        mids = by_sentence.get(sent_idx)
        if not mids:
            continue
        mids.sort(key=lambda m: mention_tokens[m][0])
        if mention_budget <= 0:
            omitted_mentions += len(mids)
            continue
        if len(mids) > mention_budget:
            omitted_mentions += len(mids) - mention_budget
            mids = mids[:mention_budget]
        mention_budget -= len(mids)
        lines.append(f"S{sent_idx + 1}")

        for mid in mids:
            own = frozenset(mention_tokens[mid])
            root = _span_root(sdoc, mention_tokens[mid])
            bits = [f"  {_mention_label(mid, mentions_map)}", f"lemma={root.lemma_}", f"pos={root.pos_}"]
            morph = _morph(root)
            if morph:
                bits.append(morph)
            bits.append(f"dep={root.dep_}")
            bits.append("head=-" if root.dep_ == "ROOT" or root.head.i == root.i
                        else f"head={_token_label(root.head.i, sdoc, tok2mention, mentions_map)}")
            lines.append(" ".join(bits))
            if lvl in ("args", "paths"):
                args = _args_of(root, own, sdoc, tok2mention, mentions_map)
                if args:
                    lines.append(f"    args: {args}")

        if lvl in ("args", "paths"):
            conn = _connectives(start, end, sdoc, tok2mention, mentions_map)
            if conn:
                lines.append(f"  connectives: {conn}")

        if lvl == "paths" and paths_budget > 0:
            # Unordered pairs sorted by token position, never by pair-list order, which the
            # pair shuffles permute: the block must be a pure function of the document.
            in_sentence = set(mids)
            seen: set = set()
            pairs: List[Tuple[str, str]] = []
            for e1, e2 in pair_ids:
                if e1 in in_sentence and e2 in in_sentence and e1 != e2:
                    key = tuple(sorted((e1, e2)))
                    if key not in seen:
                        seen.add(key)
                        pairs.append(key)
            pairs.sort(key=lambda p: (mention_tokens[p[0]][0], mention_tokens[p[1]][0]))
            rendered: List[str] = []
            for e1, e2 in pairs:
                if len(rendered) >= _MAX_PATHS_PER_SENTENCE or len(rendered) >= paths_budget:
                    break
                a, b = _span_root(sdoc, mention_tokens[e1]), _span_root(sdoc, mention_tokens[e2])
                path = _dep_path(sdoc, a, b, start, end)
                if path and len(path) <= _MAX_PATH_LEN:
                    rendered.append("    " + _format_path(
                        _mention_label(e1, mentions_map), path, sdoc, tok2mention, mentions_map))
            if rendered:
                lines.append("  paths:")
                lines.extend(rendered)
                paths_budget -= len(rendered)
                omitted = len(pairs) - len(rendered)
                if omitted > 0:
                    lines.append(f"    ({omitted} more pair(s) in this sentence omitted)")

    if omitted_mentions:
        lines.append(f"({omitted_mentions} further mention(s) not annotated)")

    if "pos" in comps:
        rows = _pos_text(view, sdoc)
        if rows:
            # Last on purpose: largest and most redundant layer, so the cap trims it first.
            lines.append("### POS (token/TAG, Penn Treebank; plain document text)")
            lines.extend(rows)

    if len(lines) <= 1:
        return ""
    block = "\n".join(lines)
    if len(block) > _MAX_BLOCK_CHARS:
        cut = block.rfind("\n", 0, _MAX_BLOCK_CHARS)
        block = block[:cut if cut > 0 else _MAX_BLOCK_CHARS] + "\n(annotation truncated)"
    return block


# ── Public API ────────────────────────────────────────────────────────────────

def annotate(docs, level: str, discourse: Sequence[str]) -> None:
    """Set `syntax_block` on every doc, in place. Never raises for a missing dependency, a
    non-English document or a malformed one: those get an empty block."""
    comps = tuple(c for c in DISCOURSE_ORDER if c in (discourse or []))
    for doc in docs:
        doc.syntax_block = ""
    if (level == "off" and not comps) or not docs:
        return

    skipped: Dict[str, int] = {}
    for d in docs:
        key = _lang_key(d.lang)
        if key not in _ENGLISH:
            skipped[key] = skipped.get(key, 0) + 1
    if skipped:
        detail = ", ".join(f"{k} {v}" for k, v in sorted(skipped.items()))
        _warn_once("skip-lang", f"[syntax] no pipeline for non-English documents ({detail}) "
                                f"— they get no annotation block")
    parseable = [d for d in docs if _lang_key(d.lang) in _ENGLISH and d.tokens and d.sentences
                 and all(t and t.strip() for t in d.tokens)]

    annotated = 0
    with _LOCK:
        nlp = _load_pipeline(with_ner="participants" in comps)
        if nlp is None:
            return
        for doc in parseable:
            try:
                sdoc, starts, ends = build_parse(nlp, doc.tokens)
                block = render(_aligned_view(doc, sdoc, starts, ends), sdoc, level, comps)
            except Exception as exc:  # one bad document must not end a run
                _warn_once(f"render:{doc.id}", f"[syntax] could not annotate doc {doc.id}: "
                                               f"{type(exc).__name__}: {exc}")
                continue
            doc.syntax_block = block
            annotated += bool(block)
    detail = f"level={level}" + (f" discourse={'+'.join(comps)}" if comps else "")
    summary = f"[syntax] {detail} | annotated {annotated}/{len(docs)} docs"
    if skipped:
        summary += f" | skipped {sum(skipped.values())} non-English"
    print(summary)
