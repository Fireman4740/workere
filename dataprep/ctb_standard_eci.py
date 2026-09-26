#!/usr/bin/env python3
"""Causal-TimeBank -> the standard-eci dataset, from source, the way the literature reads it.

Replaces agentere/dataprep/CausalTimeBank_dataprep_standard_eci.py followed by
benchmark/dataprep_llm_format.py (text, annots). That pipeline read the TimeML files; this one
reads the CAT release, as every ECI paper with a released CTB reader does (KeAGN, DICP, HOTECI,
DisfusECI, Man et al., CPATT, PTSGM).

Sources (github.com/paramitamirza/Causal-TimeBank, zips extracted at --ctb-root):
  CAT/<doc>.xml       tokens (with sentence ids), EVENTs, C-SIGNALs, CLINKs
  TimeML/<doc>.tml    only to check that each CAT event is the TimeML event instance of that number

Direction
---------
A CLINK's source is the cause and its target the effect (Mirza et al. 2014; TimeML
eventInstanceID -> relatedToEventInstance). The earlier dataprep already had this right, and so
does this one: relations.causes = [[source, target]].

What changes from the published dataset (all found by --check-against)
------------------------------------------------------------------------
  tokens      CAT's tokenization (punctuation split off: "Asia." -> "Asia", ".") instead of the
              whitespace-split TimeML text. Mention surfaces were already clean (TimeML tags the
              event word alone); what changes is the words around them. In 3 documents the old
              reader also dropped a C-SIGNAL's words from a nested EVENT on ("in punishment
              for" -> "in").
  mentions    all 6,811 EVENTs, named ei<n> as before. +4: the 3 EVENTs nested in a C-SIGNAL, and
              ei4 of wsj_0144, which has no MAKEINSTANCE in TimeML (named by the corpus's rule:
              every TimeML instance id is "ei" + its event number).
  pair_list   +32 ordered pairs, all involving those 4 mentions; every other pair is unchanged.
  unchanged   sentences (TimeML's lines are CAT's sentences: 2,654 = 2,654, same mentions in each),
              relations (the same 298 intra-sentence CLINKs, same direction), ids, document order.

Columns
-------
Standard-eci: id, doc_idx, tokens, mentions, spans, relations {"causes", "causedby"}, sentences,
pair_list, text, annots. Relations are intra-sentence only, as in every CTB result in the
literature (Mirza 2014 onwards).
Added (event-event; CTB has no coreference, and TLINKs are temporal, so left out):
  clinks   every CLINK as annotated, intra- and inter-sentence (318): source (cause), target
           (effect), signal (the C-SIGNAL's words, "" if none), intra (same sentence)

Usage
-----
  python dataprep/ctb_standard_eci.py --ctb-root ../litt/Causal-TimeBank \
      --out data/CausalTimeBank-standard-eci-workere --check-against Nofing/CausalTimeBank-standard-eci
  Publishing: dataprep/publish.py (builds and pushes all three datasets).
"""
from __future__ import annotations

import argparse
import glob
import os
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from typing import List, Optional

from standard_eci import card_configs, with_text, write


NAME = "CausalTimeBank-standard-eci"   # published as <account>/<NAME>-workere by publish.py


def read_cat(path: str):
    """tokens, sentences, events {CAT id: token positions}, signals {id: words}, CLINKs [(source, target, signal id)]."""
    root = ET.parse(path).getroot()
    toks = sorted(root.iter("token"), key=lambda t: int(t.get("id")))
    tokens = [t.text or "" for t in toks]
    pos = {t.get("id"): i for i, t in enumerate(toks)}
    by_sentence = defaultdict(list)
    for i, t in enumerate(toks):
        by_sentence[int(t.get("sentence"))].append(i)
    sentences = [[v[0], v[-1] + 1] for _, v in sorted(by_sentence.items())]
    assert all(v == list(range(v[0], v[-1] + 1)) for v in by_sentence.values()), f"{path}: a sentence is not contiguous"

    def anchored(el):
        return sorted(pos[a.get("id")] for a in el.findall("token_anchor"))

    marks = root.find("Markables")
    events = {e.get("id"): anchored(e) for e in marks.findall("EVENT")}
    signals = {s.get("id"): " ".join(tokens[i] for i in anchored(s)) for s in marks.findall("C-SIGNAL")}
    clinks = [(c.find("source").get("id"), c.find("target").get("id"), c.get("c-signalID") or "")
              for c in root.find("Relations").findall("CLINK")]
    return root.get("doc_name"), tokens, sentences, events, signals, clinks


def instance_ids(tml: str) -> dict:
    """TimeML event id number -> (event instance id, the event's words)."""
    root = ET.parse(tml).getroot()
    words = {el.get("eid"): "".join(el.itertext()).split() for el in root.find("TEXT").iter("EVENT")}
    return {m.get("eventID")[1:]: (m.get("eiid"), words[m.get("eventID")]) for m in root.findall("MAKEINSTANCE")}


def build_rows(ctb_root: str) -> List[dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(ctb_root, "CAT", "*.xml"))):
        doc_id, tokens, sentences, events, signals, clinks = read_cat(path)
        inst = instance_ids(os.path.join(ctb_root, "TimeML", doc_id + ".tml"))
        for n, (eiid, words) in inst.items():   # CAT event n is TimeML event e<n>, instance ei<n>
            assert eiid == f"ei{n}" and "".join(words) == "".join(tokens[i] for i in events[n]), (doc_id, n)

        order = sorted(events, key=lambda n: events[n][0])
        mentions = [f"ei{n}" for n in order]
        spans = [events[n] for n in order]
        idx = {n: i for i, n in enumerate(order)}
        sent_of_token = {i: s for s, (a, b) in enumerate(sentences) for i in range(a, b)}
        sent = [sent_of_token[sp[0]] for sp in spans]

        links = {"source": [], "target": [], "signal": [], "intra": []}
        causes = []
        for src, tgt, sig in clinks:
            c, e = idx[src], idx[tgt]
            links["source"].append(c)
            links["target"].append(e)
            links["signal"].append(signals[sig] if sig else "")
            links["intra"].append(sent[c] == sent[e])
            if sent[c] == sent[e]:
                causes.append([c, e])
        assert len({frozenset(p) for p in causes}) == len(causes), f"{doc_id}: a pair is linked twice"

        pair_list = [[i, j] for i in range(len(mentions)) for j in range(len(mentions))
                     if i != j and sent[i] == sent[j]]
        rows.append(with_text({
            "id": doc_id, "doc_idx": len(rows), "tokens": tokens, "mentions": mentions, "spans": spans,
            "relations": {"causes": causes, "causedby": [[e, c] for c, e in causes]},
            "sentences": sentences, "pair_list": pair_list, "clinks": links,
        }))
    return rows


def features():
    from datasets import Features, Sequence, Value
    ints = Sequence(Sequence(Value("int64")))
    return Features({
        "id": Value("string"), "doc_idx": Value("int64"),
        "tokens": Sequence(Value("string")), "mentions": Sequence(Value("string")), "spans": ints,
        "relations": {"causes": ints, "causedby": ints}, "sentences": ints, "pair_list": ints,
        "text": Value("string"), "annots": Value("string"),
        "clinks": {"source": Sequence(Value("int64")), "target": Sequence(Value("int64")),
                   "signal": Sequence(Value("string")), "intra": Sequence(Value("bool"))},
    })


def check_against(repo: str, rows: List[dict]) -> Counter:
    """What differs from the published dataset. Asserts that the documents, their order and the
    causal pairs (as mention ids, with direction) are the same, and that no mention is lost."""
    from datasets import load_dataset
    published = list(load_dataset(repo, split="train"))
    assert [p["id"] for p in published] == [r["id"] for r in rows], "documents or their order differ"
    c = Counter()
    for pub, new in zip(published, rows):
        def pairs(r):
            return [(r["mentions"][a], r["mentions"][b]) for a, b in r["relations"]["causes"]]
        assert pairs(pub) == pairs(new), f"{new['id']}: causal pairs differ"
        assert set(pub["mentions"]) <= set(new["mentions"]), f"{new['id']}: a published mention is missing"
        c["causal pairs identical (ids and direction)"] += len(pairs(new))
        c["mentions added"] += len(new["mentions"]) - len(pub["mentions"])
        c["documents retokenized"] += pub["tokens"] != new["tokens"]
        c["documents whose old reading dropped words"] += len("".join(pub["tokens"])) < len("".join(new["tokens"]))
        c["sentences, published"] += len(pub["sentences"])
        c["sentences, CAT"] += len(new["sentences"])
        c["candidate pairs (unordered), published"] += len(pub["pair_list"]) // 2
        c["candidate pairs (unordered), CAT"] += len(new["pair_list"]) // 2
    return c


CARD = """---
license: other
language:
- en
pretty_name: Causal-TimeBank — standard ECI (CAT tokenization)
tags:
- event-causality-identification
- relation-extraction
{configs}---

# Causal-TimeBank — standard ECI

Causal-TimeBank (Mirza et al. 2014) for event causality identification, in the standard-eci format,
built from the CAT release as the ECI literature reads it (KeAGN, DICP, HOTECI, DisfusECI, CPATT,
PTSGM). It supersedes `Nofing/CausalTimeBank-standard-eci`: the causal pairs are the same
({pairs} intra-sentence CLINKs, same direction), and so are the sentences. Tokens now come from CAT
(punctuation split off) instead of whitespace-split TimeML lines, and no event or word is lost: 4
EVENTs the old reader skipped are back (3 nested in a C-SIGNAL, whose following words were dropped too).

## Direction

A CLINK's source is the cause and its target the effect. `relations.causes` holds `[cause, effect]`,
`relations.causedby` the mirror `[effect, cause]`; `annots` states the same.

## Split

One split, `train` ({docs} documents): CTB has no official split. The literature uses 10-fold
cross-validation over documents, each paper with its own shuffle; workere folds by `doc_idx % 10`.

## Columns

| column | content |
|---|---|
| `id`, `doc_idx` | document identity; `doc_idx` follows the file names |
| `tokens`, `sentences` | CAT tokens; `[start, end)` token range per CAT sentence |
| `mentions`, `spans` | the {mentions:,} EVENTs, named by their TimeML event instance (`ei<n>`), and their token indices |
| `relations` | intra-sentence CLINKs: `causes: [[cause, effect]]`, `causedby: [[effect, cause]]`, indices into `mentions` |
| `pair_list` | every ordered pair of mentions in the same sentence (the candidate pairs) |
| `text`, `annots` | tokens with `<ei<n> surface>` markup; the relations as `<a ...> causes <b ...>` items |
| `clinks` | every CLINK as annotated, intra- and inter-sentence: `source` (cause), `target` (effect), `signal` (the C-SIGNAL's words, empty if none), `intra` |

## Statistics

| | |
|---|---|
| documents / events | {docs} / {mentions:,} |
| CLINKs: intra / inter-sentence | {pairs} / {inter} |
| CLINKs with a C-SIGNAL | {signalled} |
| candidate intra-sentence pairs (unordered) | {candidates:,} |

These are SemDI's figures (Li et al. 2024: 183 documents, 9,721 pairs, 298 causal). The "7,608
intra-sentence event pairs" most papers repeat (RichGCN, PPAT, CLECI, LKCER) is not reproducible
from the release and no released reader produces it. Their positive counts are 318 (all CLINKs)
or, for PPAT, 300 intra-sentence ones: 298 here and in every CAT reader.

`ei4` of `wsj_0144` ("stemming") has no MAKEINSTANCE in TimeML; CAT keeps it, and it is named by
the corpus's rule (every TimeML instance id is `ei` + its event number). It is in no CLINK.

## Build

`python dataprep/ctb_standard_eci.py --ctb-root <Causal-TimeBank clone> --out <dir>` (workere).

## Citation

Mirza, P., Sprugnoli, R., Tonelli, S. and Speranza, M. (2014). Annotating Causality in the
TempEval-3 Corpus. EACL 2014 Workshop on Computational Approaches to Causality in Language (CAtoCL).
Source: https://github.com/paramitamirza/Causal-TimeBank, which states no license; its text is
TimeBank's, as distributed in TempEval-3 (TBAQ-cleaned).
"""


def build(ctb_root: str, out: str, check_against_repo: Optional[str] = None) -> None:
    """Build from the CTB clone at `ctb_root` and write <out>/data/train-*.parquet and <out>/README.md."""
    rows = build_rows(ctb_root)
    if check_against_repo:
        for k, v in check_against(check_against_repo, rows).items():
            print(f"{v:6d}  {k}")
    s = Counter(docs=len(rows))
    for r in rows:
        s["mentions"] += len(r["mentions"])
        s["pairs"] += len(r["relations"]["causes"])
        s["inter"] += r["clinks"]["intra"].count(False)
        s["signalled"] += sum(1 for x in r["clinks"]["signal"] if x)
        s["candidates"] += len(r["pair_list"]) // 2
    for k, v in s.items():
        print(f"{v:6d}  {k}")
    write(out, {"train": rows}, features(), CARD.format(configs=card_configs(["train"]), **s))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ctb-root", required=True, help="Causal-TimeBank clone with CAT/ and TimeML/ extracted")
    ap.add_argument("--out", required=True, help="output folder (parquet under <out>/data/)")
    ap.add_argument("--check-against", default=None, help="published repo to compare with")
    args = ap.parse_args()
    build(args.ctb_root, args.out, args.check_against)


if __name__ == "__main__":
    main()
