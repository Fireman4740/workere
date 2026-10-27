# Next agent: integrate and run the A/R pilot (repo access required)

You are the implementation agent for the `workere` Event Causality Identification project. You have access to the actual repository and experiment infrastructure. The attached offline package `eci_alias_next` is a tested prototype, not an integrated pipeline. **Do not make paid model calls before verifying prerequisites and obtaining budget approval.**

## Why this experiment

The research report for virtual event mention handles proposes comparing A (distinct IDs referring to exact same mention) against R (the same repeated records with unchanged IDs). A has no gold relation information. The project's best ESC system is the historic 7-pass Luna x d1 fusion (48.64 F1 on TEST-S40). The new A/R q-vote is not the same score as the historic fusion and needs a rerun B to support baseline comparisons. The agent-6 report identifies H1 exposure of 88/88 reserve document IDs. This is not clean confirmation material.

## Mandatory first steps

1. Inspect `eci/data.py`, `eci/voting.py`, `dataprep/esc_standard_eci.py`, `prompts/esl_standard_eci_lean_coref.yaml`, `pipelines/jev/state_builder.py` and actual DEV44 split manifests. Verify exact schemas, token index base, event ID shape and candidate universe. Do not expose gold, annotations, coreref gold or relations to the model.
2. Integrate the included allowlisted `Document`, `prepare_view`, `project_one_generation` and `aggregate` functionality, or adapt it with **equivalent unit tests**. Preserve tagged document bytes and mention spans. Refuse ambiguous offsets or candidate changes.
3. Build per-document A/R views with up to one alias for at most min(16,ceil(M/4)) deterministic mentions. Project to original IDs before per-generation voting. Treat duplicate edges and reverse labels as one undirected positive per generation; keep all negative/error traces separate.
4. Use the same seven historical demos (out of topic), model version, decoding, input/output schema and request settings per paired arm. Use no new d1 calls. The historical Luna output schema may differ; rerun the control and preserve backward-compatible outputs separately.
5. Confirm exact model-tokenizer lengths. A/R auxiliary records must differ only in handle values, and full inputs must be appraised at token-level. Enforce input <=16,384 and completion <=4,096 where the model/API supports these limits; verify no truncation and record actual input/output/reasoning usage. Never claim strictly equal reasoning compute when it is not observable.
6. Write model responses as JSONL `{doc_id,arm,pass_index,status,response}` and provider usage records required by `eci_alias.budget`; run the gate before reading gold labels.
7. Before spending, prepare a concise deployment report: actual token/cost forecast from model tariffs, number of docs, queries, frozen prompts/demos, split status, smoke budget, rollback and stop conditions. Obtain explicit operator budget approval.

## Required test strategy

- Reproduce local unit tests, then integration tests for Unicode, nested and discontinuous spans, tag remapping and ID names with punctuation, accidental alias-to-alias loops, original candidate preservation and abnormal model outputs.
- One document A/R smoke (14 planned calls); inspect output parser and projection without choosing the document from known false negatives.
- Only if smoke passes, 44-document A/R on development, with 7 passes each (616 calls). Fixed threshold 4/7 for initial A/R mechanism contrast, no cherry-picking scores or prompts.
- After predictions frozen, evaluate true/missed additions vs R, F1, cost, variance across topics, and proof of actual zero-vote recovery (requires *historic* Luna vote cache, not merely this rerun B).
- If alias behavior does not beat repeated-view behavior at comparable cost, close the track. A positive exploratory result still requires a new independent confirmatory split.

## Deliverables

- Working integration branch/PR and full diff.
- Frozen run manifest with commit SHA, provider/model version, exact prompt/decode/demo hashes and budget/actual token counts.
- Unit/integration tests and source/projection audit.
- Pilot logs and a short evidence-calibrated experimental result. Explicitly separate **measured** results from hypothetical and synthetic fixtures.
