# Dataprep: CTB and MAVEN-ERE (ESC in ESC_DATAPREP_AUDIT.md)

2026-09-26. The three standard-eci datasets are now built from source by `workere/dataprep/`,
replacing agentere's `*_dataprep_standard_eci.py` + `benchmark/dataprep_llm_format.py`. Each
builder compares itself with the published dataset (`--check-against`) and is pinned by tests.
`dataprep/publish.py --token hf_... --account <account>` rebuilds all three and pushes each to
`<account>/<name>-workere` (`--only`, `--private`, `--dry-run`). Nothing is pushed yet.

| dataset | builder | published dataset vs rebuild | added columns | output |
|---|---|---|---|---|
| ESC v0.9 | `esc_standard_eci.py` | 2,893 of 5,602 pairs reversed (fixed) | `plot_links`, `causal_origin`, `causal_frame`, `coref_clusters`, `coref_instance` | `data/EventStoryLine-0.9-standard-eci-workere/` |
| CTB | `ctb_standard_eci.py` | pairs and direction right; 4 events and 3 documents' words lost; non-literature tokenization | `clinks` (all 318, with C-SIGNAL) | `data/CausalTimeBank-standard-eci-workere/` |
| MAVEN-ERE | `maven_ere_standard_eci.py` | right: identical on every column, 3,623 / 3,623 documents | `causal_type`, `coref_clusters`, `subevents`, `title`, `mention_type`, `mention_source_id` | `data/MAVEN-ERE-standard-eci-workere/` |

Shared `text` / `annots` / writing / publishing code: `dataprep/standard_eci.py` (ESC's output is
byte-identical after moving onto it).

## Themes

| dataset | theme in the hub dataset? | in the source? |
|---|---|---|
| ESC (all `0.9-*`) | yes: `topic_id` (22 ECB+ topics; dev = 37, 41) | yes (folder = topic) |
| ESC `1.5-*` | no column; the id's prefix is the topic (`1_6ecbplus.xml` = topic 1) | yes |
| CTB | no | no: TimeBank news, no topic annotation |
| MAVEN-ERE (`-standard-eci`, `-span`, `-json`) | no | only the Wikipedia article `title`, now a column in v2. No topic field (MAVEN-ERE's `type` is per event, not per document) |
| MECI | no column; ids carry one (`economic_crisis-week4-...`) | — (not rebuilt) |

## CTB

### Published dataset vs the literature

| aspect | literature | published (`Nofing/CausalTimeBank-standard-eci`) | v2 |
|---|---|---|---|
| source read | CAT release (KeAGN, DICP, HOTECI, DisfusECI, Man et al., CPATT, PTSGM) | TimeML, split on whitespace and newlines | CAT |
| documents | 183 (most papers write 184; PPAT, SemDI 183) | 183 | 183 |
| events | 6,811 (release README) | 6,807 | 6,811 |
| direction | CLINK source = cause (Mirza et al. 2014; every reader) | right | right: 318 / 318 agree with TimeML |
| gold | intra-sentence CLINKs | 298 | the same 298 (ids and direction) |
| inter-sentence CLINKs (20) | dropped | dropped | kept in `clinks`, outside `relations` |
| candidate pairs | 9,721 (SemDI); "7,608" (RichGCN, PPAT, CLECI, LKCER: not reproducible, no released reader gives it) | 9,705 | **9,721** |
| sentences | CAT sentence ids | TimeML lines: 2,654 | 2,654, the same (TimeML's lines are CAT's sentences) |
| tokens | CAT (punctuation split) | whitespace (`"Asia."`) | CAT |
| split | no official split; 10-fold CV over documents, each paper its own shuffle (DICP: seed 6688; Man et al.) | one `train` split | same; workere folds by `doc_idx % 10` |

### What the old reader lost

| document | lost | cause |
|---|---|---|
| APW19980322.0749 | "punishment for" (of the C-SIGNAL "in punishment for"), and event `ei23` | an EVENT nested in a C-SIGNAL: the reader kept the C-SIGNAL's leading words (`child.text`) and skipped everything from the nested element on |
| wsj_0568 | "said, is why" (of "That, he said, is why"), and `ei89` | same |
| wsj_1039 | "result of" (of "the result of"), and `ei16` | same |
| wsj_0144 | `ei4` (stemming) | no MAKEINSTANCE in TimeML; CAT has it. Named by the corpus rule (every instance id is `ei` + event number, 6,810 / 6,810) |

None of the 4 is in a CLINK: the gold is unchanged, the candidate set gains 16 pairs.

### Checks (`tests/test_ctb_dataprep.py`, 4 tests, pass)

| check | result |
|---|---|
| 183 docs, 6,811 events, 318 CLINKs (298 intra), 169 with a C-SIGNAL, 9,721 candidate pairs | pass |
| every CLINK equals TimeML's `eventInstanceID -> relatedToEventInstance` | 318 / 318 |
| every CAT event is the TimeML instance `ei<n>`, same words | 6,810 / 6,810 (asserted in the builder) |
| vs published: same documents and order, 298 pairs identical, +4 mentions, same sentence count | pass |

## MAVEN-ERE

The published dataset was right. `PRECONDITION` was already in `relations.causes`, and it is most of it:

| split | documents | mentions | causal pairs | CAUSE | PRECONDITION | coreference chains (>= 2) | subevent pairs |
|---|---|---|---|---|---|---|---|
| train | 2,621 | 66,325 | 47,674 | 12,376 | 35,298 (74%) | 2,222 | 17,754 |
| dev | 292 | 7,614 | 5,684 | 1,913 | 3,771 (66%) | 252 | 2,347 |
| test (= `valid.jsonl`) | 710 | 17,780 | 13,624 | 3,277 | 10,347 (76%) | 582 | 6,099 |

What v2 adds is the distinction: `causal_type` per pair. In the source, no pair has both types, and
no pair is causal in both directions (checked on all 3,623 documents).

| aspect | literature / official code | published | v2 |
|---|---|---|---|
| gold | event-pair relations expanded to every mention pair of the two chains (`causal/src/data.py`) | same | same: agrees with the official expansion, pair for pair and type for type, on all 710 test documents |
| direction | `[A, B]` = A CAUSE / PRECONDITION B | right | right |
| candidates | all n(n-1) ordered mention pairs | no `pair_list` (the loader uses all pairs) | same |
| split | train / valid; test labels hidden (CodaLab). Papers evaluate on valid or a split of it (PPAT: halves) | train.jsonl -> 90/10 train/dev (seed 42); valid.jsonl -> test | same (reproduced exactly) |
| TIMEX, temporal relations | — | excluded | excluded (you asked for no time relations) |

Added columns: `causal_type` (CAUSE / PRECONDITION per pair), `coref_clusters` (chains of >= 2
mentions), `subevents` (`[[parent, child]]`, the fourth MAVEN-ERE relation type; drop it if you
want causal + coreference only), `title`, `mention_type` (MAVEN event type), `mention_source_id`
(back to the source and to CodaLab's format).

### Checks (`tests/test_maven_dataprep.py`, 4 tests, pass)

| check | result |
|---|---|
| splits 2,621 / 292 / 710; test 17,780 mentions, 13,624 pairs | pass |
| every standard column equals `Nofing/MAVEN-ERE-standard-eci` | 3,623 / 3,623 documents |
| pairs and types equal the official baseline's expansion | 710 / 710 documents |
| layers consistent (mirror, parallel lengths, disjoint chains) | pass |

## Using them

| step | ESC | CTB | MAVEN-ERE |
|---|---|---|---|
| publish | `publish.py ... --only esc` -> `<account>/EventStoryLine-0.9-standard-eci-workere` | `--only ctb` -> `<account>/CausalTimeBank-standard-eci-workere` | `--only maven` -> `<account>/MAVEN-ERE-standard-eci-workere` |
| point workere at it | `datasets.<name>.repo_id`: the `-workere` repo, or the local folder (loads as is; checked) | same | same |
| effect on runs | directed gold changes (2,893 pairs); demos re-synthesized | `text` changes (tokens): CTB CoT demos re-synthesized once; gold unchanged | none: standard columns identical |

Sources: `litt/SOURCES.md` lists every repo and paper used, with what each says about data preparation.
