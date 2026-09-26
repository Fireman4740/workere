# ESC (EventStoryLine v0.9) dataprep audit

2026-09-26. Checked: `Nofing/EventStoryLine-0.9-standard-eci` (what agentere and workere
evaluate on) and `agentere/dataprep/EventStoryLine_dataprep_standard_eci.py`, against the ESC
source corpus (`agentere/data/EventStoryLine` = github.com/tommasoc80/EventStoryLine), the ESC
and Gao et al. papers, and the ESC readers in `litt/` (HOTECI, DisfusECI, Code4CPATT, PTSGM,
SERE, Dr.ECI) and `litt/papers/`. Nothing in agentere was modified.

**Verdict: everything is right except the direction of FALLING_ACTION links — 2,893 of the
5,602 causal pairs (51.6%) point the wrong way. The coreference extension is correct and is the
literature's. A normalized dataprep rebuilds it from source, corrected (section 5); not pushed yet.**

## 1. Aspect by aspect

| aspect | literature | published dataset | |
|---|---|---|---|
| version | v0.9 (Gao et al. 2019 and every ECI paper since) | v0.9 | ok |
| documents / split | 258 docs, 22 topics; last two topics (37, 41) = dev; 5-fold CV on the other 20 | `train` 233 docs (20 topics), `dev` 25 docs (37, 41) | ok |
| event mentions | 5,334 (all `ACTION_*` + `NEG_ACTION_*`) | 5,334 | ok |
| gold source | `evaluation_format/full_corpus/v0.9/event_mentions_extended/*.tab` — PLOT_LINKs propagated over ECB+ event coreference: 1,770 intra + 3,855 inter = 5,625 lines (Gao 2019, DiffusECI, iLIF, CaSR) | these files; undirected causal pairs identical to the source in **258/258** documents | ok |
| `null` lines (23: 7 intra, 16 inter) | split: not causal in HOTECI, DiffusECI, Dr.ECI, ManHieu; causal in Gao's own counts (1,770 / 3,855 include them), PPAT, ICCL, DAPrompt, GLM4ECI, SemDI, CPATT, PTSGM | excluded (no direction) | ok; v2 keeps them in `unlabelled_pairs` |
| intra positives on the 233 CV docs | Dr.ECI / SERE main table: 1,592 pairs (3,184 ordered) | 1,592 pairs — agentere's `binary_intra` gold+ = 3,184 | ok |
| intra candidate pairs, 233 docs | Dr.ECI: 8,999 unordered (17,998 ordered) | 8,999 | ok |
| endpoints | — | every `.tab` endpoint is exactly one ESC mention span (0 unmapped) | ok |
| docs without a `.tab` | — | 5 (1_17, 1_6, 14_10, 32_7, 4_6): they have 0 PLOT_LINKs, so the raw-XML fallback never fires | ok |
| **direction** | PRECONDITION(a, b) = a causes b; FALLING_ACTION(a, b) = **b causes a** | both read as a causes b | **wrong for 2,893 pairs** |
| evaluation | undirected existence (Gao 2019: "we do not further distinguish if A causes B vs. B causes A"); direction as an extra setting (iLIF) | agentere `binary*` = existence on ordered rows (unaffected); `micro/macro` = directed (affected) | see 5 |

## 2. Direction: every source agrees, except the dataprep

| source | what it says | FALLING_ACTION(a, b) |
|---|---|---|
| Caselli & Vossen 2017 (the ESC paper) | "rising action, events which are circumstantial to, cause or enable another event, or falling action, which explicitly mark speculations and consequences, i.e. events which are the (anticipated) outcome or the effect of another event"; "PLOT_LINK relations are asymmetrical"; example "earthquake *rising action* killed" | a is the effect of b |
| raw v0.9 XML, the annotators' explicit-causality flags | PRECONDITION + `CAUSES=TRUE` 70 vs `CAUSED_BY=TRUE` 10; FALLING_ACTION + `CAUSED_BY=TRUE` 35 vs `CAUSES=TRUE` 2 | b causes a |
| ESC `create_gold_document.py` (builds the `.tab`) | writes each pair in ascending token order and flips PRECONDITION <-> FALLING_ACTION when it swaps the ends: the two labels are inverses | b causes a |
| independent re-propagation (this audit) | from raw PLOT_LINKs + ECB+ chains, with FALLING_ACTION = target causes source: reproduces all 5,602 `.tab` pairs, **5,602/5,602 directions agree** | b causes a |
| Gao et al. 2019 (NAACL, the standard setting) | rising / falling action "indicate the directions of causal relations and intuitively correspond to 'precondition' and 'consequence'" | a is the consequence |
| HOTECI (`data_modules/datasets.py`) | `'FALLING_ACTION': 'is caused by'`, `'PRECONDITION': 'causes'` | b causes a |
| CPATT / PTSGM (`makedataset-eventstoryline.py`) | PRECONDITION: "event1 is the cause of event2", else swapped | b causes a |
| Dr.ECI released ESC gold (`data/ESC/intra_sent_causality.json`) | each pair twice: PRECONDITION with label 1 (e1 -> e2), FALLING_ACTION with label 2 (e2 -> e1), 1,592 each | b causes a |
| iLIF (Liu et al. 2024) | "we designate event pairs annotated with 'PRECONDITION' as CAUSE relation, and 'FALLING_ACTION' as EFFECT relation" | b causes a |
| DisfusECI, SERE | binary: both labels are one causal class | — |
| **agentere dataprep** (lines 162-163) | `causes.append([src_idx, tgt_idx])` for both relTypes | **a causes b** |

## 3. Caveats inherited from ESC itself (kept, to stay aligned with the literature)

| | count | note |
|---|---|---|
| pairs the propagation asserts in both directions | 276 of 5,602 (4.9%) | two annotated links meet on the same mention pair through coreference; ESC's builder keeps the last one written, and so does every copy of the data (Dr.ECI included). A directed metric cannot be right on both. |
| re-running ESC's own propagation on the released XML | +63 causal pairs, 12 causal pairs overwritten by a colliding `null` link (last write wins) | the released `.tab` files differ slightly from a fresh run; every paper's 1,770 / 3,855 are the released files, so they stay the reference |
| propagated pairs dropped by ESC's builder | 230 | both ends start at the same token (overlapping mentions) |
| Gao's candidate-pair counts (7,805 intra / 46,521 cross) | Gao drops aspectual, causative, perception and reporting mentions (639 of 5,334): exactly 7,805 intra / 46,251 cross (the paper's 46,521 is a digit swap). All mentions: 10,347 / 60,232 | two candidate conventions (section 6); ours is the all-mentions one (ERGO, PPAT, HOTECI, Dr.ECI). v2's `mention_type` gives Gao's |
| agentere's `binary` counts ordered rows | — | equals the literature's unordered-pair score when the model emits both directions (the prompts require it); `analyze.py sweep --undirected` gives the unordered score |
| `text` / `annots` columns | — | produced by `benchmark/dataprep_llm_format.py`, not by agentere; the 30 discontinuous mentions are tagged piece by piece under one id (`<11 in> a <11 wheelchair>`) |

## 4. What the published direction error has touched

| affected | not affected |
|---|---|
| directed `micro_*` / `macro_f1` / `per_label` of every ESC run | every `binary*` number, every sweep, every comparison with the literature (all undirected) |
| the ESC few-shot demos: `format_gold_output` shows the gold directed, so about half of each demo's links (and its synthesized CoT) taught the reverse direction | coreference propagation (the undirected pair set is right) |
| `rule_sets.esl` and any transitive closure on ESC | MAVEN (direction accuracy 99.1%), CTB |

## 5. The fix: one normalized dataprep

`workere/dataprep/esc_standard_eci.py` builds the dataset from source (ESC XML + the released
`.tab` gold + ECB+) in one step, replacing agentere's `EventStoryLine_dataprep_standard_eci.py`
followed by `benchmark/dataprep_llm_format.py` (which added `text` / `annots`). Run with
`--check-against Nofing/EventStoryLine-0.9-standard-eci`, it proves its provenance: read the old
way, it reproduces the published dataset exactly (258/258 documents, all 11 columns); read
correctly, it differs in the direction of 2,893 pairs and nothing else.

Added columns (event-event layers the standard-eci format dropped; no temporal or location links):
`plot_links` (the PLOT_LINKs as annotated: rel_type, CAUSES / CAUSED_BY flags, signal),
`causal_origin` (annotated 2,229 / coreference copy 3,373), `causal_frame` (the annotated frame of
the originating link: PRECONDITION 2,283 / FALLING_ACTION 2,413 / both 906), `coref_clusters`
(646 ECB+ event clusters), `coref_instance` (2,963 mentions with an ECB+ cross-document instance),
`mention_type` (the ESC markable type; gives Gao's candidate set), `unlabelled_pairs` (the 23 `null` lines).
Output with its dataset card: `workere/data/EventStoryLine-0.9-standard-eci-workere/`.

| check (`tests/test_esc_dataprep.py`) | result |
|---|---|
| 258 docs (233 / 25), dev topics {37, 41}, 5,334 mentions, 5,602 pairs | pass |
| agrees with Dr.ECI's directed intra gold | 3,184 / 3,184 (published: 1,408) |
| agrees with the annotators' CAUSES / CAUSED_BY flags | > 85% (the raw flags themselves contradict in a few links) |
| legacy reading == published dataset, byte for byte | pass |
| workere loader | 233 docs, 10,118 gold triples |

Re-scoring existing ESC runs against the corrected gold (their predictions unchanged):

| run | binary F1 | directed micro P/R/F1, published gold | corrected gold | direction accuracy on found links |
|---|---|---|---|---|
| 21f411ff | 33.9 | 22.6 / 13.6 / 17.0 | 41.5 / 24.9 / 31.1 | 92.0% |
| d46be8a3 | 35.4 | 23.3 / 14.4 / 17.8 | 43.0 / 26.5 / 32.8 | 92.5% |
| 85dccc90 (whitelist) | 48.4 | 33.3 / 19.0 / 24.2 | 61.9 / 35.2 / 44.9 | 92.7% |
| 5b80d245 (workere) | 35.2 | 26.6 / 13.7 / 18.1 | 47.7 / 24.5 / 32.4 | 92.1% |

Using it:

| step | |
|---|---|
| publish | `python dataprep/publish.py --token hf_... --account <account> --only esc` -> `<account>/EventStoryLine-0.9-standard-eci-workere` (rebuilds, then uploads the folder: parquet + card) |
| point workere at it | `datasets.event_story_line.repo_id: "<account>/EventStoryLine-0.9-standard-eci-workere"` (or the local folder) |
| CoT demos | the demo gold output changes, so ESC CoT cache keys change: demos are re-synthesized once, now teaching the right direction |

## 6. Against the wider literature

Checked 2026-09-26 against every ESC reader and every processed copy of ESC in `litt/` (the
readers of section 2, plus the repos added since). Processed data was compared pair by pair.

| source | what was compared | gold source | `null` lines | direction | agrees with v2 |
|---|---|---|---|---|---|
| ERGO (COLING 2022), cached data | 258 docs, all `.tab` lines | extended `.tab` | kept as lines | binary | 5,602 / 5,602 pairs, all 5,334 mentions |
| PPAT (IJCAI 2023), released folds | 258 docs | extended `.tab` | causal | binary (pairs in `.tab` order) | 5,625 / 5,625 (= 5,602 + 23 `null`), all 5,334 mentions |
| SemDI (EMNLP 2024), released data | 7,805 intra pairs, 253 docs | extended `.tab` | causal | binary | 7,791 / 7,805 labels; the rest are the 7 `null` pairs and 7 pairs whose surface is ambiguous in my mapping |
| Gao et al. 2023 (ChatGPT causal reasoning), released sample | 1,000 directed pairs | extended `.tab` | — | FALLING_ACTION = e2 -> e1 | 1,000 / 1,000, directions included |
| GLM4ECI (ACL 2025, directional ECI), code | label mapping | ICCL/DAPrompt rows | causal (falls through to label 1) | PRECONDITION: e1 = cause; FALLING_ACTION: e1 = effect | same direction rule |
| Man et al., released intra data (ManHieu) | 8,102 intra records | raw PLOT_LINKs, **not propagated** | not causal | FALLING_ACTION(A, B) = B causes A (code comment) | direction 1,564 / 1,588; misses 137 propagated pairs (a deviation of theirs) |
| ICCL, DAPrompt (code) | reader | extended `.tab`, Gao's mention set | causal (`0 if NONE else 1`) | binary | Gao's convention |
| KeAGN, KnowMMR (code) | reader | extended `.tab` (KnowMMR's released script points at v1.0) | — | binary | — |
| KnowledgeFusedECI (no paper) | reader | raw PLOT_LINKs, sampled negatives | — | tags source `<CAUSE>`, target `<EFFECT>` whatever the relType (agentere's mistake), but the label keeps the relType | not the standard setting |
| RichGCN's ESC copy | `annotated_data/v0.9` | — | — | — | 258 / 258 files byte-identical to ours |

Nothing in the literature reads FALLING_ACTION as agentere's dataprep did. Where the literature
splits, v2 now carries both sides:

| convention | side A | side B | v2 |
|---|---|---|---|
| candidate mentions | all 5,334: ERGO, PPAT, HOTECI, DiffusECI, Dr.ECI, SERE, KnowMMR | Gao's 4,695 (no aspectual / causative / perception / reporting): Gao 2019, ICCL, DAPrompt, GLM4ECI, SemDI | A; `mention_type` gives B |
| `null` lines | not causal: HOTECI, DiffusECI, Dr.ECI, ManHieu | causal: Gao's counts, PPAT, ICCL, DAPrompt, GLM4ECI, SemDI, CPATT, PTSGM | A in `relations` (they have no direction); `unlabelled_pairs` gives B |

What the choice is worth, on run 5b80d245 (233 docs, intra-sentence, undirected):

| scoring | P | R | F1 |
|---|---|---|---|
| v2 as is | 55.2 | 51.7 | 53.4 |
| + `null` lines causal | 55.3 | 51.8 | 53.5 |
| Gao's setting (his mentions, `null` causal, unreachable gold kept) | 57.5 | 51.2 | 54.2 |

+0.8 F1, inside the run-to-run noise (sd ≈ 1.2): Gao's set removes 66 false positives on reporting
and similar mentions. Worth stating when comparing with a Gao-lineage number, not worth changing
the dataset for.

## Sources

- ESC repository and gold builder: https://github.com/tommasoc80/EventStoryLine (`create_gold_document.py`, `annotated_data/v0.9`, `evaluation_format/full_corpus/v0.9`)
- Caselli & Vossen 2017, The Event StoryLine Corpus: https://aclanthology.org/W17-2711.pdf
- Gao, Choubey & Huang 2019, Modeling Document-level Causal Structures for Event Causal Relation Identification: https://people.engr.tamu.edu/huangrh/papers/naacl19_causal_relation_identification.pdf
- `litt/HOTECI/data_modules/{readers,datasets}.py`, `litt/DisfusECI/diffus_ie/data_modules/{data_reader,constants}.py`, `litt/Code4CPATT/code/Event Causality Identification/makedataset-eventstoryline.py`, `litt/PTSGM/code/event_storyline/makedataset-eventstoryline.py`, `litt/SERE/CPATT_dataset_format.ipynb`, `litt/Dr.ECI/data/ESC/intra_sent_causality.json`
- `litt/papers/`: iLIF (Liu et al. 2024), DiffusECI (Man et al. 2024), ERGO (Chen et al. 2022), CaSR (Zhao et al. 2026), SERE (Hao et al. 2026)
