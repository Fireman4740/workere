#!/usr/bin/env python3
"""EventStoryLine v0.9 -> the standard-eci dataset, from source, in one script.

Replaces the two steps that produced Nofing/EventStoryLine-0.9-standard-eci —
agentere/dataprep/EventStoryLine_dataprep_standard_eci.py (tokens, mentions, relations) followed
by benchmark/dataprep_llm_format.py (text, annots) — with one builder that gets the direction of
the causal relations right, and adds the ESC annotation layers the standard-eci format dropped.

Sources (github.com/tommasoc80/EventStoryLine, cloned at --esl-root):
  annotated_data/v0.9/<topic>/<doc>.xml.xml                     tokens, event mentions, PLOT_LINKs
  evaluation_format/full_corpus/v0.9/event_mentions_extended/   the gold: PLOT_LINKs propagated over
                                                                ECB+ event coreference (5,625 lines:
                                                                1,770 intra / 3,855 inter, as papers count)
  ECB+_LREC2014/ECB+/<topic>/<doc>.xml                          event coreference (ECB+)

Direction
---------
ESC's builder (create_gold_document.py) writes every pair of the gold in ascending token order and
flips PRECONDITION <-> FALLING_ACTION when that swaps the two ends. So a gold line reads

    a  b  PRECONDITION     a causes b        (a is a precondition of b)
    a  b  FALLING_ACTION   b causes a        (a is a falling action — a consequence — of b)

The earlier dataprep read both as "a causes b", reversing 2,893 of the 5,602 causal pairs (51.6%).
See workere/ESC_DATAPREP_AUDIT.md for the evidence (ESC paper, the annotators' CAUSES/CAUSED_BY
flags, create_gold_document.py, Gao et al. 2019, HOTECI, CPATT, Dr.ECI's released gold, iLIF).
`null` gold lines (23 PLOT_LINKs without a relType) have no direction, so they are not in relations;
the literature splits on them (HOTECI, DiffusECI, Dr.ECI: not causal; Gao's counts, PPAT, ICCL,
DAPrompt, GLM4ECI, SemDI, CPATT: causal). They are kept apart, in unlabelled_pairs.

Columns
-------
Standard-eci (same values as the published dataset, direction corrected):
  id, doc_idx, topic_id, split ("train" = the 20 cross-validation topics, "dev" = topics 37, 41),
  tokens, mentions (ESC m_ids, event mentions only), spans, sentences ([start, end) token ranges),
  relations {"causes": [[cause, effect]], "causedby": [[effect, cause]]} (indices into mentions),
  text (tokens with <m_id surface> markup), annots ("<a ...> causes <b ...>, " items)
Added (event-event annotation that standard-eci dropped; indices into mentions):
  plot_links      the PLOT_LINKs exactly as the annotators drew them, before propagation:
                  source, target, rel_type (PRECONDITION | FALLING_ACTION | "" when unset),
                  causes / caused_by (the explicit-causality flags), signal
  causal_origin   parallel to relations["causes"]: "annotated" when a PLOT_LINK joins exactly these
                  two mentions, "coreference" when the pair is a propagated copy
  causal_frame    parallel to relations["causes"]: how the annotators framed the link it comes from —
                  "PRECONDITION" (drawn cause -> effect), "FALLING_ACTION" (drawn effect -> cause),
                  or "both" when links of both frames produce it
  coref_clusters  ECB+ event coreference within the document (clusters of >= 2 mentions)
  coref_instance  parallel to mentions: the ECB+ cross-document event instance id ("" if none);
                  mentions of different documents sharing an id refer to the same real-world event
  mention_type    parallel to mentions: the ESC markable (ACTION_OCCURRENCE, NEG_ACTION_REPORTING, ...)
  unlabelled_pairs  the gold lines whose PLOT_LINK has no relType (ESC's "null", 23): not in relations
                  (they have no direction), but causal in the papers' 1,770 / 3,855 counts

The literature's two candidate sets
-----------------------------------
All 5,334 mentions (ERGO, PPAT, HOTECI, DiffusECI, Dr.ECI, SERE): 10,347 intra-sentence pairs.
Gao et al. (2019) and those who follow them (ICCL, DAPrompt, GLM4ECI, SemDI) drop the aspectual,
causative, perception and reporting mentions (639, see `gao_candidate`): 7,805 intra / 46,251 cross
pairs (the paper prints 46,521), and the 23 gold pairs touching a dropped mention become unreachable.

Usage
-----
  python dataprep/esc_standard_eci.py --esl-root ../agentere/data/EventStoryLine \
      --out data/EventStoryLine-0.9-standard-eci-workere \
      --check-against Nofing/EventStoryLine-0.9-standard-eci      # prove the provenance
  Publishing: dataprep/publish.py (builds and pushes all three datasets).
"""
from __future__ import annotations

import argparse
import glob
import os
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

from standard_eci import card_configs, with_text, write

DEV_TOPICS = {37, 41}
# ACTION_* and NEG_ACTION_* markables: the 5,334 event mentions every paper counts.
EVENT_PREFIXES = ("ACTION", "NEG_ACTION")
NAME = "EventStoryLine-0.9-standard-eci"   # published as <account>/<NAME>-workere by publish.py
CAUSAL = ("PRECONDITION", "FALLING_ACTION")
# Gao et al. (2019): "we ... exclude aspectual, causative, perception and reporting event mentions".
GAO_EXCLUDED = ("ASPECTUAL", "CAUSATIVE", "PERCEPTION", "REPORTING")


def gao_candidate(mention_type: str) -> bool:
    """Whether a mention of this type is in Gao et al.'s candidate set."""
    return not mention_type.endswith(GAO_EXCLUDED)


# ── source readers ────────────────────────────────────────────────────────────

def read_esc(path: str):
    """tokens, sentences, event mentions (sorted by first token), the raw PLOT_LINKs, and the
    mentions' markable types."""
    root = ET.parse(path).getroot()
    tokens, tid_pos, sent_of_pos = [], {}, {}
    for tok in sorted(root.findall(".//token"), key=lambda t: int(t.get("t_id", 0))):
        pos = len(tokens)
        tokens.append(tok.text or "")
        tid_pos[int(tok.get("t_id", 0))] = pos
        sent_of_pos[pos] = int(tok.get("sentence", 0))
    buckets = defaultdict(list)
    for pos, sid in sent_of_pos.items():
        buckets[sid].append(pos)
    sentences = [[min(v), max(v) + 1] for _, v in sorted(buckets.items())]

    raw = []
    for m in root.findall(".//Markables/*"):
        if not m.tag.startswith(EVENT_PREFIXES) or not m.get("m_id"):
            continue
        anchors = [(a.get("t_id"), tid_pos[int(a.get("t_id"))]) for a in m.findall("token_anchor")
                   if a.get("t_id") and int(a.get("t_id")) in tid_pos]
        if anchors:
            raw.append((min(p for _, p in anchors), m.get("m_id"), sorted(p for _, p in anchors),
                        frozenset(t for t, _ in anchors), m.tag))
    raw.sort(key=lambda x: x[0])           # stable: equal first tokens keep markable order
    mentions = [r[1] for r in raw]
    spans = [r[2] for r in raw]
    by_tokens = {r[3]: i for i, r in enumerate(raw)}   # {t_id, ...} -> mention index (order-free)
    idx = {m: i for i, m in enumerate(mentions)}

    links = []
    for pl in root.iter("PLOT_LINK"):
        s, t = pl.find("source"), pl.find("target")
        if s is None or t is None or s.get("m_id") not in idx or t.get("m_id") not in idx:
            continue           # one v0.9 link has a participant, not an event, as its source
        links.append({"source": idx[s.get("m_id")], "target": idx[t.get("m_id")],
                      "rel_type": pl.get("relType") or "", "causes": pl.get("CAUSES") == "TRUE",
                      "caused_by": pl.get("CAUSED_BY") == "TRUE", "signal": pl.get("SIGNAL") or ""})
    types = [r[4] for r in raw]
    return root.get("doc_name") or os.path.basename(path), tokens, sentences, mentions, spans, by_tokens, links, types


def read_gold(tab: Optional[str], by_tokens: Dict[frozenset, int]) -> List[Tuple[int, int, str]]:
    """The .tab lines as (first, second, relType), in file order."""
    out = []
    if tab and os.path.exists(tab):
        with open(tab, encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 3:
                out.append((by_tokens[frozenset(parts[0].split("_"))],
                            by_tokens[frozenset(parts[1].split("_"))], parts[2]))
    return out


def read_coref(ecb_path: str, by_tokens: Dict[frozenset, int]):
    """ECB+ event coreference: {target: [mention indices]} and {mention index: instance id}."""
    root = ET.parse(ecb_path).getroot()
    marks = {el.get("m_id"): el for el in root.find("Markables")}
    tokens_of = {m: frozenset(a.get("t_id") for a in el.findall("token_anchor"))
                 for m, el in marks.items() if el.tag.startswith(EVENT_PREFIXES) and el.findall("token_anchor")}
    groups, instance = defaultdict(list), {}
    for rel in root.find("Relations"):
        target = rel.find("target").get("m_id")
        for src in rel.findall("source"):
            i = by_tokens.get(tokens_of.get(src.get("m_id"), frozenset()))
            if i is None:
                continue       # an entity chain, or an ECB+ event ESC did not keep
            groups[target].append(i)
            if rel.tag == "CROSS_DOC_COREF":
                instance[i] = marks[target].get("instance_id") or ""
    return groups, instance


# ── derived layers ────────────────────────────────────────────────────────────

def cause_effect(first: int, second: int, rel: str) -> Tuple[int, int]:
    return (first, second) if rel == "PRECONDITION" else (second, first)


def origins(links, clusters_of: Dict[int, List[int]]) -> Dict[Tuple[int, int], List[Tuple[str, bool]]]:
    """(cause, effect) -> [(frame, is the annotated pair itself)] for every propagated copy of
    every causal PLOT_LINK — the expansion create_gold_document.py performs."""
    out = defaultdict(list)
    for ln in links:
        if ln["rel_type"] not in CAUSAL:
            continue
        c, e = cause_effect(ln["source"], ln["target"], ln["rel_type"])
        for x in clusters_of.get(c, [c]):
            for y in clusters_of.get(e, [e]):
                if x != y:
                    out[(x, y)].append((ln["rel_type"], {x, y} == {c, e}))
    return out


# ── build ─────────────────────────────────────────────────────────────────────

def build_rows(esl_root: str, legacy_direction: bool = False) -> List[dict]:
    """One row per document, sorted by (topic, file name); doc_idx contiguous over all of them.
    `legacy_direction` reproduces the published (reversed) direction — used only by the check."""
    xml_dir = os.path.join(esl_root, "annotated_data", "v0.9")
    tab_dir = os.path.join(esl_root, "evaluation_format", "full_corpus", "v0.9", "event_mentions_extended")
    ecb_dir = os.path.join(esl_root, "ECB+_LREC2014", "ECB+")
    paths = sorted(glob.glob(os.path.join(xml_dir, "*", "*.xml.xml")),
                   key=lambda p: (int(os.path.basename(os.path.dirname(p))), os.path.basename(p)))
    rows = []
    for path in paths:
        topic = int(os.path.basename(os.path.dirname(path)))
        doc_id, tokens, sentences, mentions, spans, by_tokens, links, types = read_esc(path)
        if len(mentions) < 2:
            continue
        stem = os.path.basename(path)[: -len(".xml.xml")]
        gold = read_gold(os.path.join(tab_dir, str(topic), stem + ".tab"), by_tokens)
        groups, instance = read_coref(os.path.join(ecb_dir, str(topic), doc_id), by_tokens)
        clusters = sorted(sorted(set(g)) for g in groups.values() if len(set(g)) >= 2)
        clusters_of = {i: c for c in clusters for i in c}
        made = origins(links, clusters_of)

        causes, origin, frame, unlabelled = [], [], [], []
        for a, b, rel in gold:
            if rel not in CAUSAL:
                assert rel == "null", f"{doc_id}: unknown gold relType {rel!r}"
                unlabelled.append([a, b])
                continue
            c, e = (a, b) if legacy_direction else cause_effect(a, b, rel)
            causes.append([c, e])
            got = made.get(cause_effect(a, b, rel), [])
            if not got:  # every gold pair must come from an annotated link, directly or by coreference
                raise ValueError(f"{doc_id}: gold pair {mentions[a]},{mentions[b]} has no originating PLOT_LINK")
            frames = {f for f, _ in got}
            origin.append("annotated" if any(self for _, self in got) else "coreference")
            frame.append("both" if len(frames) > 1 else next(iter(frames)))
        rows.append(with_text({
            "id": doc_id, "doc_idx": len(rows), "topic_id": topic,
            "split": "dev" if topic in DEV_TOPICS else "train",
            "tokens": tokens, "mentions": mentions, "spans": spans,
            "relations": {"causes": causes, "causedby": [[e, c] for c, e in causes]},
            "sentences": sentences,
            "plot_links": {k: [ln[k] for ln in links] for k in ("source", "target", "rel_type", "causes", "caused_by", "signal")},
            "causal_origin": origin, "causal_frame": frame,
            "coref_clusters": clusters,
            "coref_instance": [instance.get(i, "") for i in range(len(mentions))],
            "mention_type": types, "unlabelled_pairs": unlabelled,
        }))
    return rows


def features():
    from datasets import Features, Sequence, Value
    ints = Sequence(Sequence(Value("int64")))
    return Features({
        "id": Value("string"), "doc_idx": Value("int64"), "topic_id": Value("int64"), "split": Value("string"),
        "tokens": Sequence(Value("string")), "mentions": Sequence(Value("string")), "spans": ints,
        "relations": {"causes": ints, "causedby": ints}, "sentences": ints,
        "text": Value("string"), "annots": Value("string"),
        "plot_links": {"source": Sequence(Value("int64")), "target": Sequence(Value("int64")),
                       "rel_type": Sequence(Value("string")), "causes": Sequence(Value("bool")),
                       "caused_by": Sequence(Value("bool")), "signal": Sequence(Value("string"))},
        "causal_origin": Sequence(Value("string")), "causal_frame": Sequence(Value("string")),
        "coref_clusters": ints, "coref_instance": Sequence(Value("string")),
        "mention_type": Sequence(Value("string")), "unlabelled_pairs": ints,
    })


STANDARD = ("id", "doc_idx", "topic_id", "split", "tokens", "mentions", "spans", "relations", "sentences", "text", "annots")


def check_against(repo: str, legacy_rows: List[dict], rows: List[dict]) -> Counter:
    """The legacy-direction build must equal the published dataset on every standard column, and
    the corrected build must differ from it in direction only."""
    from datasets import load_dataset
    published = {r["id"]: r for split in ("train", "dev") for r in load_dataset(repo, split=split)}
    c = Counter()
    assert set(published) == {r["id"] for r in rows}, "document sets differ"
    for old, new in zip(legacy_rows, rows):
        pub = published[old["id"]]
        for k in STANDARD:
            assert pub[k] == old[k], f"{old['id']}: column {k} differs from {repo}"
        assert {frozenset(p) for p in pub["relations"]["causes"]} == {frozenset(p) for p in new["relations"]["causes"]}
        for p, q in zip(pub["relations"]["causes"], new["relations"]["causes"]):
            c["pairs kept" if p == q else "pairs reversed"] += 1
        c["documents identical to the published standard columns (legacy direction)"] += 1
    return c


CARD = """---
license: cc-by-3.0
language:
- en
pretty_name: EventStoryLine v0.9 — standard ECI, direction-corrected
tags:
- event-causality-identification
- relation-extraction
- coreference
{configs}---

# EventStoryLine v0.9 — standard ECI (direction-corrected)

The EventStoryLine Corpus (ESC) v0.9 for event causality identification, in the standard-eci
format, built from the ESC and ECB+ sources. It supersedes `Nofing/EventStoryLine-0.9-standard-eci`,
which is identical on every shared column except that **{reversed:,} of its {pairs:,} causal pairs
({reversed_pct:.1f}%) point the wrong way**: it read every gold line `a b FALLING_ACTION` as "a causes b".

## Direction

The gold is ESC's coreference-extended evaluation format
(`evaluation_format/full_corpus/v0.9/event_mentions_extended`), whose builder writes each pair in
token order and flips the label when it swaps the ends:

| gold line | meaning |
|---|---|
| `a b PRECONDITION` | a causes b |
| `a b FALLING_ACTION` | b causes a (a is a consequence of b) |

As in Caselli & Vossen (2017), Gao et al. (2019), HOTECI, CPATT, Dr.ECI and iLIF. `relations.causes`
holds `[cause, effect]` and `relations.causedby` the mirror `[effect, cause]`; `annots` states the same.

## Splits

| split | documents | topics |
|---|---|---|
| train | {docs_train} | the 20 topics used for 5-fold cross-validation |
| dev | {docs_dev} | 37, 41 (the last two topics: the standard development set) |

The gold has the counts every paper reports: 1,770 intra- and 3,855 inter-sentence lines over the
258 documents (23 of them `null` PLOT_LINKs, in `unlabelled_pairs`), {mentions:,} event mentions. On the 233
train documents, 1,592 intra-sentence causal pairs — the gold of Dr.ECI and SERE.

## Columns

| column | content |
|---|---|
| `id`, `doc_idx`, `topic_id`, `split` | document identity; `doc_idx` is stable, sorted by (topic, file name) |
| `tokens`, `sentences` | the document's tokens; `[start, end)` token range per sentence |
| `mentions`, `spans` | event mentions (ESC `m_id`, all `ACTION_*` / `NEG_ACTION_*`) and their token indices |
| `relations` | `causes: [[cause, effect]]`, `causedby: [[effect, cause]]`, indices into `mentions` |
| `text`, `annots` | tokens with `<m_id surface>` markup; the relations as `<a ...> causes <b ...>` items |
| `plot_links` | the PLOT_LINKs exactly as annotated, before propagation: `source`, `target`, `rel_type` (PRECONDITION / FALLING_ACTION / empty), the explicit-causality flags `causes`, `caused_by`, and `signal` |
| `causal_origin` | per `relations.causes` entry: `annotated` (a PLOT_LINK joins these two mentions) or `coreference` (a propagated copy) |
| `causal_frame` | per `relations.causes` entry: how the originating link was drawn — `PRECONDITION` (cause -> effect), `FALLING_ACTION` (effect -> cause), or `both` |
| `coref_clusters` | ECB+ event coreference within the document (clusters of >= 2 mentions) |
| `coref_instance` | per mention: the ECB+ cross-document event instance (empty if none) |
| `mention_type` | per mention: the ESC markable type (`ACTION_OCCURRENCE`, `NEG_ACTION_REPORTING`, ...) |
| `unlabelled_pairs` | the {unlabelled} gold lines whose PLOT_LINK has no relType (ESC's `null`), as `[first, second]`: not in `relations` (no direction), but counted as causal in the papers' 1,770 / 3,855 |

## Candidate pairs: the literature's two conventions

| | mentions | intra-sentence pairs | used by |
|---|---|---|---|
| all event mentions | {mentions:,} | {intra_all:,} | ERGO, PPAT, HOTECI, DiffusECI, Dr.ECI, SERE |
| Gao et al. (2019): without aspectual, causative, perception, reporting mentions | {mentions_gao:,} | {intra_gao:,} | Gao 2019, ICCL, DAPrompt, GLM4ECI, SemDI |

Gao's set is `mention_type` not ending in `ASPECTUAL`, `CAUSATIVE`, `PERCEPTION` or `REPORTING`
({cross_gao:,} cross-sentence pairs; the paper prints 46,521). Gold pairs touching a dropped mention
stay in the gold there and cannot be found.

## Statistics

| | |
|---|---|
| causal pairs (each in both directions) | {pairs:,} |
| annotated / coreference copies | {annotated:,} / {coreference:,} |
| frame PRECONDITION / FALLING_ACTION / both | {frame_p:,} / {frame_f:,} / {frame_b:,} |
| annotated PLOT_LINKs | {plot_links:,} |
| coreference clusters / mentions with a cross-document instance | {clusters:,} / {instances:,} |

Caveats inherited from ESC's release, kept for comparability: 276 pairs are reached in both
directions by propagation and ESC keeps one; the release differs slightly from re-running its own
propagation script (63 pairs, 12 overwritten by `null` links).

## Build

`python dataprep/esc_standard_eci.py --esl-root <EventStoryLine clone> --out <dir>` (workere).

## Citation

Caselli, T. and Vossen, P. (2017). The Event StoryLine Corpus: A New Benchmark for Causal and
Temporal Relation Extraction. Events and Stories in the News Workshop. —
Cybulska, A. and Vossen, P. (2014). Using a sledgehammer to crack a nut? Lexical diversity and
event coreference resolution. LREC. Both corpora: CC BY 3.0.
"""


def build(esl_root: str, out: str, check_against_repo: Optional[str] = None) -> None:
    """Build from the ESC clone at `esl_root` and write <out>/data/*.parquet and <out>/README.md."""
    rows = build_rows(esl_root)
    legacy = build_rows(esl_root, legacy_direction=True)
    if check_against_repo:
        for k, v in check_against(check_against_repo, legacy, rows).items():
            print(f"{v:6d}  {k}")

    s = Counter()
    for r, old in zip(rows, legacy):
        s[f"docs_{r['split']}"] += 1
        s["mentions"] += len(r["mentions"])
        s["pairs"] += len(r["relations"]["causes"])
        s["reversed"] += sum(p != q for p, q in zip(r["relations"]["causes"], old["relations"]["causes"]))
        s["annotated"] += r["causal_origin"].count("annotated")
        s["coreference"] += r["causal_origin"].count("coreference")
        s["frame_p"] += r["causal_frame"].count("PRECONDITION")
        s["frame_f"] += r["causal_frame"].count("FALLING_ACTION")
        s["frame_b"] += r["causal_frame"].count("both")
        s["plot_links"] += len(r["plot_links"]["source"])
        s["clusters"] += len(r["coref_clusters"])
        s["instances"] += sum(1 for x in r["coref_instance"] if x)
        s["unlabelled"] += len(r["unlabelled_pairs"])
        s["mentions_gao"] += sum(map(gao_candidate, r["mention_type"]))
        for name, keep in (("all", lambda t: True), ("gao", gao_candidate)):
            by_sentence = Counter(next(k for k, (a, b) in enumerate(r["sentences"]) if a <= sp[0] < b)
                                  for sp, t in zip(r["spans"], r["mention_type"]) if keep(t))
            n = sum(by_sentence.values())
            s[f"intra_{name}"] += sum(v * (v - 1) // 2 for v in by_sentence.values())
            s[f"cross_{name}"] += n * (n - 1) // 2 - sum(v * (v - 1) // 2 for v in by_sentence.values())
    for k, v in s.items():
        print(f"{v:6d}  {k}")
    splits = ("train", "dev")
    write(out, {sp: [r for r in rows if r["split"] == sp] for sp in splits}, features(),
          CARD.format(configs=card_configs(splits), reversed_pct=100 * s["reversed"] / s["pairs"], **s))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--esl-root", required=True, help="clone of github.com/tommasoc80/EventStoryLine")
    ap.add_argument("--out", required=True, help="output folder (parquet under <out>/data/)")
    ap.add_argument("--check-against", default=None, help="published repo to prove the provenance against")
    args = ap.parse_args()
    build(args.esl_root, args.out, args.check_against)


if __name__ == "__main__":
    main()
