#!/usr/bin/env python3
"""MAVEN-ERE -> the standard-eci dataset, from source, with the layers standard-eci dropped.

Replaces agentere/dataprep/Maven-ERE_dataprep_standard_eci.py followed by
benchmark/dataprep_llm_format.py (text, annots). Their output is reproduced exactly (checked by
--check-against); what this adds is the annotation they flattened away.

Source (github.com/THU-KEG/MAVEN-ERE, data release extracted at --data-dir):
  train.jsonl   2,913 documents          -> train (90%) / dev (10%), the published split
  valid.jsonl     710 documents          -> test (test.jsonl's labels are hidden, CodaLab only)

MAVEN-ERE annotates relations between events, i.e. coreference chains; a mention-level pair is
causal when its two chains are (the official baseline's reading, causal/src/data.py). Both causal
types are causal: [A, B] under CAUSE or PRECONDITION means A -> B. No pair carries both types,
none is causal both ways.

Columns
-------
Standard-eci (identical to Nofing/MAVEN-ERE-standard-eci):
  id, doc_idx, tokens, mentions (e<i>, by position), spans, sentences,
  relations {"causes": [[cause, effect]], "causedby": [[effect, cause]]} (CAUSE and PRECONDITION),
  text, annots
Added (event-event; no TIMEX, no temporal relations):
  title              the Wikipedia article's title (MAVEN-ERE has no topic field; this is the nearest)
  causal_type        parallel to relations["causes"]: "CAUSE" | "PRECONDITION"
  coref_clusters     the coreference chains of >= 2 mentions (indices into mentions)
  subevents          [[parent, child]]: child is a subevent of parent (chains expanded to mentions)
  mention_type       per mention: its event type (MAVEN's 168 types, e.g. "Attack")
  mention_source_id  per mention: MAVEN-ERE's mention id (to map back to the source / CodaLab)

Usage
-----
  python dataprep/maven_ere_standard_eci.py --data-dir ../litt/MAVEN-ERE/data/MAVEN_ERE \
      --out data/MAVEN-ERE-standard-eci-workere --check-against Nofing/MAVEN-ERE-standard-eci
  Publishing: dataprep/publish.py (builds and pushes all three datasets).
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Optional

from standard_eci import card_configs, with_text, write

NAME = "MAVEN-ERE-standard-eci"   # published as <account>/<NAME>-workere by publish.py
CAUSAL = ("CAUSE", "PRECONDITION")   # the order the published dataset lists them in


def build_doc(raw: dict, doc_idx: int) -> dict:
    tokens, starts = [], []
    for sent in raw["tokens"]:
        starts.append(len(tokens))
        tokens.extend(sent)
    sentences = [[s, e] for s, e in zip(starts, starts[1:] + [len(tokens)]) if s < e]

    raw_mentions = [(m, ev) for ev in raw["events"] for m in ev["mention"]]
    raw_mentions.sort(key=lambda x: (x[0]["sent_id"], x[0]["offset"][0]))   # stable, as published
    spans = [list(range(starts[m["sent_id"]] + m["offset"][0], starts[m["sent_id"]] + m["offset"][1]))
             for m, _ in raw_mentions]
    of_event: Dict[str, List[int]] = defaultdict(list)
    for i, (_, ev) in enumerate(raw_mentions):
        of_event[ev["id"]].append(i)

    def expand(pairs):
        return [[a, b] for x, y in pairs for a in of_event[x] for b in of_event[y]]

    causes, types = [], []
    for rel in CAUSAL:
        for pair in expand(raw["causal_relations"][rel]):
            causes.append(pair)
            types.append(rel)
    assert len({frozenset(p) for p in causes}) == len(causes), f"{raw['id']}: a pair is causal twice"
    return with_text({
        "id": raw["id"], "doc_idx": doc_idx, "tokens": tokens,
        "mentions": [f"e{i}" for i in range(len(raw_mentions))], "spans": spans,
        "relations": {"causes": causes, "causedby": [[e, c] for c, e in causes]},
        "sentences": sentences,
        "title": raw["title"], "causal_type": types,
        "coref_clusters": [v for v in of_event.values() if len(v) >= 2],
        "subevents": expand(raw["subevent_relations"]),
        "mention_type": [ev["type"] for _, ev in raw_mentions],
        "mention_source_id": [m["id"] for m, _ in raw_mentions],
    })


def build_splits(data_dir: str) -> Dict[str, List[dict]]:
    """train.jsonl -> train / dev by the published cut (datasets' train_test_split(test_size=0.1,
    seed=42), which depends on the row count only); valid.jsonl -> test. doc_idx runs over
    train.jsonl then valid.jsonl, before the cut."""
    from datasets import Dataset
    rows = {}
    n = 0
    for name in ("train", "valid"):
        with open(os.path.join(data_dir, f"{name}.jsonl"), encoding="utf-8") as f:
            docs = [json.loads(line) for line in f if line.strip()]
        rows[name] = [build_doc(d, n + i) for i, d in enumerate(docs)]
        n += len(docs)
    cut = Dataset.from_dict({"i": list(range(len(rows["train"])))}).train_test_split(test_size=0.1, seed=42)
    return {"train": [rows["train"][i] for i in cut["train"]["i"]],
            "dev": [rows["train"][i] for i in cut["test"]["i"]],
            "test": rows["valid"]}


def features():
    from datasets import Features, Sequence, Value
    ints = Sequence(Sequence(Value("int64")))
    strs = Sequence(Value("string"))
    return Features({
        "id": Value("string"), "doc_idx": Value("int64"), "tokens": strs, "mentions": strs, "spans": ints,
        "relations": {"causes": ints, "causedby": ints}, "sentences": ints,
        "text": Value("string"), "annots": Value("string"),
        "title": Value("string"), "causal_type": strs, "coref_clusters": ints, "subevents": ints,
        "mention_type": strs, "mention_source_id": strs,
    })


STANDARD = ("id", "doc_idx", "tokens", "mentions", "spans", "relations", "sentences", "text", "annots")


def check_against(repo: str, splits: Dict[str, List[dict]]) -> Counter:
    """Every standard column of every document must equal the published dataset's, split by split."""
    from datasets import load_dataset
    c = Counter()
    for split, rows in splits.items():
        published = list(load_dataset(repo, split=split))
        assert [p["id"] for p in published] == [r["id"] for r in rows], f"{split}: documents or their order differ"
        for pub, new in zip(published, rows):
            for k in STANDARD:
                assert pub[k] == new[k], f"{split} {new['id']}: column {k} differs from {repo}"
        c[f"{split}: documents identical on every standard column"] = len(rows)
    return c


CARD = """---
license: gpl-3.0
language:
- en
pretty_name: MAVEN-ERE — standard ECI, with causal types, coreference and subevents
tags:
- event-causality-identification
- event-relation-extraction
- coreference
{configs}---

# MAVEN-ERE — standard ECI

MAVEN-ERE (Wang et al. 2022) for event causality identification, in the standard-eci format, built
from the official release. Every column of `Nofing/MAVEN-ERE-standard-eci` is here unchanged; the
added columns keep what that format flattened: which causal type each pair has, the coreference
chains, subevents, the article title and event types.

## Causal relations

MAVEN-ERE annotates relations between events (coreference chains). A mention pair is causal when its
chains are, as in the official baseline. `relations.causes` holds `[cause, effect]` for both types —
`CAUSE` and `PRECONDITION`, `[A, B]` meaning A -> B — and `causal_type` says which, pair by pair.
`relations.causedby` is the mirror; `annots` states the same.

## Splits

| split | documents | from |
|---|---|---|
| train | {docs_train:,} | `train.jsonl`, 90% (`train_test_split(test_size=0.1, seed=42)`) |
| dev | {docs_dev:,} | `train.jsonl`, the other 10% |
| test | {docs_test:,} | `valid.jsonl` (the official test labels are hidden; papers evaluate on it too) |

## Columns

| column | content |
|---|---|
| `id`, `doc_idx`, `title` | document identity; `doc_idx` runs over train.jsonl then valid.jsonl; the article title |
| `tokens`, `sentences` | the document's tokens; `[start, end)` token range per sentence |
| `mentions`, `spans` | event mentions `e<i>` in reading order, and their token indices (TIMEX excluded) |
| `relations` | `causes: [[cause, effect]]`, `causedby: [[effect, cause]]`, indices into `mentions` |
| `text`, `annots` | tokens with `<e<i> surface>` markup; the relations as `<a ...> causes <b ...>` items |
| `causal_type` | per `relations.causes` entry: `CAUSE` or `PRECONDITION` |
| `coref_clusters` | coreference chains of >= 2 mentions |
| `subevents` | `[[parent, child]]`: the child mention's event is a subevent of the parent's |
| `mention_type` | per mention: its MAVEN event type |
| `mention_source_id` | per mention: MAVEN-ERE's own mention id |

## Statistics

| | train | dev | test |
|---|---|---|---|
| mentions | {mentions_train:,} | {mentions_dev:,} | {mentions_test:,} |
| causal pairs | {pairs_train:,} | {pairs_dev:,} | {pairs_test:,} |
| of which CAUSE / PRECONDITION | {cause_train:,} / {pre_train:,} | {cause_dev:,} / {pre_dev:,} | {cause_test:,} / {pre_test:,} |
| coreference chains (>= 2 mentions) | {clusters_train:,} | {clusters_dev:,} | {clusters_test:,} |
| subevent pairs | {sub_train:,} | {sub_dev:,} | {sub_test:,} |

## Build

`python dataprep/maven_ere_standard_eci.py --data-dir <MAVEN_ERE folder> --out <dir>` (workere).

## Citation

Wang, X., Chen, Y., Ding, N., Peng, H., Wang, Z., Lin, Y., Han, X., Hou, L., Li, J., Liu, Z., Li, P.
and Zhou, J. (2022). MAVEN-ERE: A Unified Large-scale Dataset for Event Coreference, Temporal,
Causal, and Subevent Relation Extraction. EMNLP 2022. Source: https://github.com/THU-KEG/MAVEN-ERE (GPL-3.0).
"""


def build(data_dir: str, out: str, check_against_repo: Optional[str] = None) -> None:
    """Build from MAVEN-ERE's train/valid.jsonl in `data_dir` and write <out>/data/*.parquet and <out>/README.md."""
    splits = build_splits(data_dir)
    if check_against_repo:
        for k, v in check_against(check_against_repo, splits).items():
            print(f"{v:6d}  {k}")
    s = Counter()
    for split, rows in splits.items():
        s[f"docs_{split}"] = len(rows)
        for r in rows:
            s[f"mentions_{split}"] += len(r["mentions"])
            s[f"pairs_{split}"] += len(r["relations"]["causes"])
            s[f"cause_{split}"] += r["causal_type"].count("CAUSE")
            s[f"pre_{split}"] += r["causal_type"].count("PRECONDITION")
            s[f"clusters_{split}"] += len(r["coref_clusters"])
            s[f"sub_{split}"] += len(r["subevents"])
    for k, v in s.items():
        print(f"{v:7d}  {k}")
    write(out, splits, features(), CARD.format(configs=card_configs(splits), **s))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="folder with MAVEN-ERE's train.jsonl and valid.jsonl")
    ap.add_argument("--out", required=True, help="output folder (parquet under <out>/data/)")
    ap.add_argument("--check-against", default=None, help="published repo to prove the provenance against")
    args = ap.parse_args()
    build(args.data_dir, args.out, args.check_against)


if __name__ == "__main__":
    main()
