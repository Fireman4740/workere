"""Model answer -> relation triples.

The answer is the last *populated* JSON array in the response. Reasoning often contains
brackets, quoted strings and even stray `[]`, so arrays are found by bracket depth walking back
from the last `]`, and `json.loads` is the arbiter: a slice that does not parse is skipped and
an earlier `]` tried. The scan is deliberately not string-aware — a forward scan that tracked
quotes lost the whole answer to one unpaired `"` in ordinary quoted CoT. A bare `[]` is the
answer only when no populated array exists anywhere ("no causal relations").
"""
from __future__ import annotations

import json
from typing import Dict, Iterator, List, Tuple

from eci.labels import normalize

Triple = Tuple[str, str, str]


class ParseError(ValueError):
    pass


def iter_json_arrays(text: str) -> Iterator[str]:
    """Balanced `[...]` slices of `text`, last one first."""
    end = len(text)
    while True:
        close = text.rfind("]", 0, end)
        if close == -1:
            return
        depth, open_at = 0, -1
        for i in range(close, -1, -1):
            ch = text[i]
            if ch == "]":
                depth += 1
            elif ch == "[":
                depth -= 1
                if depth == 0:
                    open_at = i
                    break
        if open_at == -1:
            # Unbalanced: a stray `]`, or one inside a string value the depth count cannot
            # see. Offer the widest few slices and let json.loads decide — bounded, so a
            # bracket-heavy reasoning block cannot make this quadratic.
            probe = close
            for _ in range(8):
                probe = text.rfind("[", 0, probe)
                if probe == -1:
                    break
                yield text[probe: close + 1]
            end = close
            continue
        yield text[open_at: close + 1]
        end = open_at


def answer_array(text: str) -> list:
    """The answer array. Raises ParseError when the response holds no JSON array at all."""
    found_any = False
    for block in iter_json_arrays(text):
        try:
            arr = json.loads(block)
        except json.JSONDecodeError:
            continue
        found_any = True
        if arr:
            return arr
    if not found_any:
        raise ParseError("no JSON array found in output")
    return []


def parse_triples(text: str, normalize_labels: bool = True) -> List[Triple]:
    """(src, label, tgt) for every `{"pair": "A,B", "label": ...}` item of the answer array.

    An item that is not an object is a malformed answer (ParseError -> the pass is retried),
    as in agentere. With `normalize_labels`, "Causes" / " causes" / "caused by" become the
    canonical label instead of splitting the pair's votes.
    """
    triples: List[Triple] = []
    for item in answer_array(text):
        if not isinstance(item, dict):
            raise ParseError(f"answer array item is not an object: {item!r:.80}")
        pair = item.get("pair", "")
        if not isinstance(pair, str) or "," not in pair:
            continue
        src, tgt = (p.strip() for p in pair.split(",", 1))
        label = item.get("label", "Unknown")
        triples.append((src, normalize(label) if normalize_labels else label, tgt))
    return triples


def parse_justifications(text: str) -> Dict[str, str]:
    """{"src,tgt": why} from the same array the labels came from; empty when the prompt asks
    for no rationale. Never raises — a missing rationale must not cost a pass that parsed."""
    out: Dict[str, str] = {}
    try:
        for block in iter_json_arrays(text):
            try:
                arr = json.loads(block)
            except json.JSONDecodeError:
                continue
            if not arr:
                continue
            for item in arr:
                if not isinstance(item, dict):
                    continue
                pair = item.get("pair", "")
                why = item.get("why") or item.get("justification") or item.get("reason")
                if isinstance(pair, str) and "," in pair and why:
                    src, tgt = (p.strip() for p in pair.split(",", 1))
                    out[f"{src},{tgt}"] = str(why).strip()
            break
    except Exception:
        return {}
    return out


def first_json_object(text: str):
    """The outermost `{...}` of a tool sub-call's answer (greedy, as the tools always did)."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ParseError("no JSON object found in output")
    return json.loads(text[start: end + 1])


def outer_json_array(text: str) -> list:
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end <= start:
        raise ParseError("no JSON array found in output")
    return json.loads(text[start: end + 1])
