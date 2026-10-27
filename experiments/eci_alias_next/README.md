# workere ECI - alias follow-up experiment (offline launch kit)

**Status as of 2026-10-27:** implementation of the *local* experiment protocol, unit tests and a synthetic end-to-end dry run are complete. **No Luna/Haiku/d1 model calls have been made. No empirical F1 gain has been measured.** This package is not a drop-in replacement for the private workere pipeline.

## Source of the plan

The user-provided agent-6 report, *Alias virtuels de mentions : redondance controlee des entrees, projection et vote* (dated 2026-10-27), recommends a narrow pilot contrasting A (new handles pointing to existing mentions) with R (same records repeated using original handles). The source reports a 14-test diagnostic of its own projector; this is an independent prototype with **37 locally passing unit tests**, not a verification of those earlier 14 tests. The source also reports overlap of **88/88** reserved document IDs with H1 jobs and summaries. That overlap could not be independently rechecked in this workspace because the source manifests were not attached.

## What is implemented

- Strict public-document normalization, with gold and annotation labels rejected from model input.
- Zero-based token-index coordinates, including discontiguous spans. Never interprets token indices as character offsets.
- Deterministic choice of up to `min(16,ceil(M/4))` original mentions by SHA-256, no gold or scores.
- Arms **B** (original), **R** (repeated identity records), **A** (new equivalent handles), **P** (bijective ID renaming of tags only).
- Complete `private_views.jsonl` and model-facing **allowlisted** `public_jobs.jsonl` for seven slots per arm/document. They are *plans*, not API requests.
- Projection from virtual handles back to original mention IDs. Strict ID/candidate checks, loop removal, one vote per original unordered pair per generation.
- Aggregation with fixed seven slots and a **preproposed** decision threshold of 4/7. Failed attempted slots count as zero; missing/unexecuted slots are a hard error.
- Gold-separated scoring of confusion counts, F1, additions/removals, utility, and exploratory topic bootstrap.
- A budget fairness gate using actual model-version, demonstration, prompt/decode hashes, token usage and truncation flags. Without real model usage this gate **cannot** pass.
- A reserve-ID intersection command, which requires the real source ID manifests.
- A synthetic pipeline fixture. Its results were intentionally fabricated to check software behavior. Never interpret its F1 as a real experiment.

## Missing dependencies for a genuine experiment

1. Access to the *workere* repository and caches for the 44-document development split.
2. The exact historical ESC prompt, the seven per-document demos from other topics and frozen Luna model/decode settings.
3. A provider adapter which takes each job's `public_input` and fills `demo_slot` with its frozen demonstration.
4. A tokenizer or provider estimate for **actual full input tokens**, and actual provider usage accounting, to ensure A and R are within the declared budget. The current `max-aux-chars` preflight is NOT an exact 768-token bound.
5. A genuinely unexposed confirmation set; the 88 IDs from the agent report cannot currently be called an untouched reserve.
6. Explicit approval to spend API credits on the proposed **616 calls** for A+R on 44 documents. This kit makes **zero paid calls**.

If any one of these prerequisites is missing, do not claim an empirical result or that the confirmatory study has started.

## Input schema

`public_docs.jsonl`: one object per document, *no labels or gold fields*:

```json
{"doc_id":"example","topic":"topic_1","text":"<e1 Rain> caused <e2 flooding>.","tokens":["Rain","caused","flooding","."],"mentions":["e1","e2"],"spans":{"e1":[0],"e2":[2]},"candidates":[["e1","e2"]]}
```

Only `doc_id,text,tokens,mentions,spans,candidates,topic` are considered. In particular, never feed an unfiltered `eci/data.py` document object into prompts: the report indicates it may contain gold fields. The `text` may be a tagged serialization; `spans` are token-indices into `tokens`, NOT character spans. All candidate pairs must be original, explicit and unordered. If original candidate extraction uses a different format, write an adapter and test it.

## Run all local tests

```bash
cd eci_alias_next
python3 -m unittest discover -s tests -v
```

## Generate experiment *jobs* without model calls

```bash
python3 -m eci_alias.cli prepare \
  --input /path/to/public_docs_DEV44.jsonl \
  --out run_alias_AR \
  --arms A R \
  --passes 7
```

Outputs:

- `run_alias_AR/public_jobs.jsonl`: model-facing, **only explicitly allowed fields**; each row must be combined by the provider adapter with the existing frozen system prompt and external demo. Never send raw `private_views.jsonl` to the model.
- `run_alias_AR/private_views.jsonl`: kept on evaluator side, includes projection and original candidates.
- `run_alias_AR/ar_checks.jsonl`: static A/R provenance and approximate character sizes.
- `run_alias_AR/STATUS.txt`: reminds that jobs have not been executed.

**Important:** The historical prompt and provider call cannot be reconstructed exactly from the uploaded report alone. This prototype's JSON output contract is deliberately explicit: a list `[[source_id, "causes" | "causedby", target_id], ...]`, or `[]`. A baseline under the changed output contract **must be rerun**. `q >= 4/7` does not preserve the old directed-score fusion threshold of `0.39`.

## Provider response format

The adapter writes one JSONL record for each actually attempted job, with a list of triples:

```json
{"doc_id":"example","arm":"A","pass_index":0,"status":"ok","response":[["v__0000","causes","e2"]]}
```

If a call genuinely failed, log `{"doc_id":"example","arm":"A","pass_index":0,"status":"error","response":null}`. Do not silently omit calls. There must be exactly 7 response rows per view. New IDs are valid only for A; P requires the changed ID mapping. Every output is projected to original unordered pairs before voting. Each generation contributes at most one vote per pair.

## Aggregate and evaluate

```bash
python3 -m eci_alias.cli aggregate \
  --views run_alias_AR/private_views.jsonl \
  --responses /path/to/actual_provider_responses.jsonl \
  --out run_alias_AR/results.jsonl
```

Only after predictions have been fixed, provide a **separate** `gold.jsonl` with `doc_id` and `positive_pairs`, and run:

```bash
python3 -m eci_alias.cli evaluate \
  --views run_alias_AR/private_views.jsonl \
  --results run_alias_AR/results.jsonl \
  --gold /path/to/gold_DEV44.jsonl \
  --out run_alias_AR/evaluation.json
```

Measure A vs R first, then B/P only if warranted. Record all new TP, FP, lost TP, lost FP and cost by document/topic. Because this is development on an exposed split, do not call this independent confirmation.

## Actual usage gate

For each A/R provider response, output `usage.jsonl` with `doc_id,arm,pass_index,input_tokens,completion_tokens,model_version,demo_id,base_prompt_sha256,decode_sha256,truncated`. `base_prompt_sha256` must hash the fixed base prompt (excluding varying auxiliary records); `decode_sha256` must hash frozen decode settings. The provider adapter, not this offline package, must produce accurate token counts.

```bash
python3 -m eci_alias.budget --usage /path/to/usage.jsonl --out budget_gate.json
```

The gate blocks missing slots, mismatched demos/model configurations, input-token deltas >1%, input >16384, completion >4096, missing usage and truncation. A pass is necessary but not sufficient for equal hidden compute, which may be unobservable.

## Reserve exposure audit

Extract only document IDs (one per line) from the true project manifests, then:

```bash
python3 -m eci_alias.cli audit-reserve \
  --reserve-ids reserve88_ids.txt \
  --h1-ids h1_manifest_ids.txt \
  --out reserve_overlap.json
```

The user-provided agent-6 report claims 88/88 overlap. This command can independently check that claim when the manifests are accessible, but it cannot establish whether gold labels were inspected.

## Included synthetic demonstration

```bash
python3 -m eci_alias.cli prepare \
  --input examples/public_docs_SYNTHETIC.jsonl \
  --out examples/out \
  --arms B R A P
python3 examples/generate_fake_responses.py \
  examples/out/private_views.jsonl examples/out/responses_SYNTHETIC.jsonl
python3 -m eci_alias.cli aggregate \
  --views examples/out/private_views.jsonl \
  --responses examples/out/responses_SYNTHETIC.jsonl \
  --out examples/out/results_SYNTHETIC.jsonl
python3 -m eci_alias.cli evaluate \
  --views examples/out/private_views.jsonl \
  --results examples/out/results_SYNTHETIC.jsonl \
  --gold examples/gold_SYNTHETIC.jsonl \
  --out examples/out/eval_SYNTHETIC.json
```

This explicitly fabricates A more positive than R. **The synthetic F1 is meaningless scientifically**; it tests only a pipeline path.

## Scientific decision rule

The first decisive contrast is **A vs R**, on the same documents and same 7 demonstrations. Candidate selection must never use gold. Publish A/R delta-F1, net TP/FP, marginal precision of additions, newly recovered baseline zero-vote pairs, topic-level dispersion, actual token costs, and limitations. An exploratory target of +0.5 F1 points over R was proposed by the agent; it is **not** a measured gain, significance threshold, or power analysis. If A and R are indistinguishable at comparable costs, stop the *specific* virtual-handle hypothesis, while keeping possible repetition effects separate.

## No accidental claim of novelty

Repeated prompts, test-time augmentation, invariance transformations and vote aggregation have published precedents. The narrower unconfirmed contribution is **new equivalent IDs for the same original occurrence, followed by projected one-vote-per-generation extraction**, distinguished from plain repetition and renamed IDs.
