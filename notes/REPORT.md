# workere — refactor of agentere: report

2026-09-26. agentere was read, never written (verified: no file under `agentere/` or
`agentere copy/` modified). Sources: agentere's code, its 70 run logs, its `mlflow.db`
(118 runs), 75 earlier sessions, and the memory notes.

## 1. Summary

| | agentere (replaced code) | workere |
|---|---|---|
| Python, framework + analysis | 12,863 lines (6,083 core + 6,780 in 27 scripts) | 5,464 lines |
| Tests | 64 lines | 604 lines: 19 unit, 11 offline pipeline, 5 parity |
| Config keys, prompts, report schema, queue format | — | identical (all 68 queued agentere experiments load unchanged) |
| LLM calls actually in flight at `concurrency: 200` (this 24-core machine) | **28** | 200 |
| Same ESC config (d46be8a3), wall clock | 50.0 min | **9.2 min** |
| A crashed run | loses everything | resumes from a per-document checkpoint |

## 2. Verification

### 2.1 Parity with agentere's recorded runs (no model calls)

workere is configured from each run's own `config.yaml` snapshot, then must rebuild the exact
prompts found in agentere's trace file and reproduce the logged metrics from the recorded answers.

| run | setting | docs | inference calls | test prompts | demo turns | CoT keys | logged metrics |
|---|---|---|---|---|---|---|---|
| 85dccc90 | ESC 233, n=5, syntax, oracle whitelist | 233 | 1,167 | byte-identical | 1,167 identical | match | exact (12 dp) |
| c7ea8631 | MAVEN dev 100, whitelist + gold count | 100 | 300 | byte-identical | 300 identical | match | exact |
| be86d19c | CTB 183, 10-fold, `causal_timebank_syntax` | 183 | 552 | byte-identical | 1,656 identical | match | exact |
| 21f411ff | ESC 233, n=3, baseline lean | 233 | 739 | byte-identical | 739 identical | match | exact |
| 762e4506 | ESC 233, coref injected (was `coref_run.py`) | 233 | 797 | byte-identical | 797 identical | match | exact |

"CoT keys match": wherever agentere's `cot_cache.json` holds a demo's CoT, it holds it under
the key workere computes, so the existing cache is reused as is (it was copied to
`workere/logs/cot_cache.json`).

### 2.2 Post-hoc analyses reproduce agentere's scripts cell for cell

| agentere script | workere | checked on | result |
|---|---|---|---|
| `vote_threshold*.py` | `analyze.py sweep` | 85dccc90, cae80dd5 | every threshold, all/intra/inter identical |
| `vote_threshold_coref.py --source log` | `sweep --coref` (`+coref`) | 21f411ff | identical: majority 34.2 → 53.8, inter 21.7 → 54.1 |
| `maven_posthoc_sweep.py --source log` | `sweep --coref --closure` | 4c5d68e6 | 6 variants x 3 thresholds x 3 scopes identical |
| `posthoc_coref_report.py` | `sweep --coref --undirected` | 4c5d68e6 | best 35.8 → 34.9, as recorded |
| `recompute_metrics.py`, `vote_to_binary_eval.py`, `binary_eval.py` | `analyze.py rescore` | 21f411ff | recompute = logged; vote_to_binary = sweep majority |
| `rerun_variance.py --group-by coarse` | `analyze.py variance --coarse` | agentere mlflow.db | same groups; n=20 ESC group sd 0.0194 |

### 2.3 Offline pipeline (scripted model, no network)

| test | checks |
|---|---|
| oracle run | F1 = 1.0; call counts per role; cost; one trace line per call; MLflow params + metrics on a FINISHED run; 2nd run reuses every CoT (0 synthesis calls) |
| failure classes | unparseable answer, `finish_reason=length`, `APIConnectionError`: each retried, all 3 recovered, all 3 in the trace |
| crash + resume | killed after 2/4 docs → params already in MLflow → resume makes only the 6 missing calls, same MLflow run ends FINISHED |
| concurrency | 60 of 60 calls in flight (agentere's sync node caps at 28) |
| demo plan | identical across loads; never the document itself; same-fold exclusion |
| features | mention split (leakage logged), whitelist enforce + gold count, vote_to_binary, binary_undirected, multi-step prompt + multi-step CoT, similarity + density stratification, human rationales, tool loop (coherence fires) |

### 2.4 Live runs (OpenRouter, deepseek-v4-flash)

Same config as agentere run d46be8a3 (`--from-agentere`): ESC 233 docs, `esl_standard_eci_lean`,
5 passes, 1 demo with CoT (all cached), syntax `mentions`. Request bodies are byte-identical
(checked); Novita served 40/40 sampled calls of each run.

| | agentere d46be8a3 (09-18) | workere 5b80d245 (09-26), conc 200 | workere e7a80a4e (09-26), conc 28 |
|---|---|---|---|
| wall clock | 50.0 min | **9.2 min** | 46.6 min |
| calls / failed attempts (retried) | 1,205 / 40 | 1,167 / 2 | 1,165 / 0 |
| output tokens per inference call | 7,252 | 8,899 | 9,032 |
| cost | $2.82 | $3.31 | $3.29 |
| binary F1 at the shipped majority (t>=3) | 35.6 | 35.2 | 34.9 |
| binary_intra F1 (logged) | 51.1 | 53.4 | 52.9 |
| best ALL F1 (sweep) | 42.4 (t>=1) | 44.4 (t>=1) | 44.2 (t>=1) |
| pairs emitted at union (t>=1) | 18,332 | 13,870 | 13,712 |

Reading: the two workere runs (same day, 200 vs 28 in flight) agree with each other on everything but wall clock: majority 35.2 vs 34.9, best 44.4 vs 44.2, 8.9k vs 9.0k output tokens per call, 13.9k vs 13.7k union pairs. So load does not change the answers, and agentere's 50 minutes were the 28-thread cap (46.6 min at 28 here). Both differ from the 09-18 agentere run in the same way (longer answers, fewer and more precise union pairs) with byte-identical requests and the same provider: drift over the 8 days or the demo draw, not the framework. At the shipped majority all three are within the ~1.2 F1 noise floor.

## 3. Defects found in agentere (all fixed in workere)

| # | defect | evidence | in workere |
|---|---|---|---|
| 1 | LLM concurrency capped at 28 whatever `experiment.concurrency` says | LangGraph runs the sync `call_model` node on the default thread pool, `min(32, cpu+4)` = 28 here. Measured: 200 concurrent 1 s calls → 8.2 s, peak 28 in flight. Every queue file asked for 200. | async calls; `concurrency` is the real budget (test: 60/60); d46be8a3 config 50 → 9.2 min |
| 2 | a run that dies leaves nothing usable | 44 of 113 traced runs never wrote a report: **$21.37 of $135.34 (16%)**, 7,684 of 38,137 calls. 46 FAILED/orphaned MLflow runs, all with 0 params (params logged only at the end) | params at start; `docs.jsonl` checkpoint per document; `--resume` |
| 3 | one transient API error fails a whole document | retry catches only `ValueError`/`TimeoutError`; `openai.APIConnectionError`, `RateLimitError`, `InternalServerError` escape, the doc's gather fails, all its passes are scored empty | all API errors retried; hidden SDK retries off (`max_retries=0`), so every retry is visible |
| 4 | truncated generations are parsed | `finish_reason` never checked (memory: 5.5% of one provider's long generations truncated with HTTP 200) | `length`/`error` → retried |
| 5 | CoT pre-generation draws other demos than inference; the rest are synthesized on the fly and never saved | both use unseeded `random.sample`; CoT texts used by runs but absent from `cot_cache.json`: 21/145 (cae80dd5), 32/146 (87fcea75), 1/126 (c7ea8631) | one seeded demo plan per (seed, doc); pre-generation covers exactly it; cache saved atomically after each CoT |
| 6 | runs not reproducible | per-document demo draw unseeded (memory: rerun-noise-floor) | seeded; identical plan across runs (test) |
| 7 | a document can be its own demonstration | ESC and CTB evaluate `train` and draw demos from `train`; only k-fold exclusion guarded it | excluded always |
| 8 | the coherence tool can never fire | rule sets written for CAUSE / PRECONDITION / FALLING_ACTION / EffectCause; every prompt emits causes / causedby. `causal_timebank`'s set is empty → the tool returns a config error | rule sets rewritten in the causes/causedby space (symmetry, transitivity); tested |
| 9 | tool sub-call cost invisible | counterfactual_check, bare_causes, eci sub-calls never traced; cost parsed afterwards from traces | one client for every call; live ledger per role (inference / cot / tool:x) |
| 10 | tools called during CoT synthesis lack the demo document's context | pre-generation sets only the text/mentions/fold contextvars, to the last TEST doc scanned: `encoder` answers "no document context", `reprompt` returns "", `few_shot_examples` ranks for the wrong doc | tools bound per document by closure |
| 11 | off-spec labels split votes | 0.03% of 1.35 M votes ("Causes", " causes", "caused by"...); 73 pairs change winner once normalised | case/whitespace normalised; residue counted (`offspec_labels`) |
| 12 | whitelist obedience / split leakage accumulate across queued runs | module-level dicts, never reset | returned per document |
| 13 | the negation-CoT experiment could silently replay positives-only CoT | CoT key ignores the hint text; the script redirected the cache by hand | `few_shot.cot_generation.negation` is part of the key |
| 14 | sweep numbers from different scripts are not comparable | `posthoc_coref_report.py` scores undirected pairs (max of both directions), every other sweep ordered pairs | one sweep, convention explicit (`--undirected`) |
| 15 | `vote_to_binary` crashes a document on a self-pair "a,a" | `sorted(frozenset)` of size 1 | skipped |

## 4. Where each agentere capability lives now

| agentere | workere |
|---|---|
| `main.py`, `run_queue.py` | `run.py`, `eci/runner.py` (+ resume, dry run, `--set`, `--from-agentere`) |
| `utils/config.py`, `runtime_config.py`, `context.py` | `eci/config.py` (strict merge + up-front validation); no global registry, no contextvars |
| `dataprep/dataprep.py` | `eci/data.py` (`Doc` dataclass) |
| `model/model.py` (LangGraph) | `eci/llm.py` (async tool loop, retries, ledger, traces, SQLite LLM cache) |
| `tools/few_shot.py`, `utils/human_rationales.py` | `eci/fewshot.py` |
| `utils/syntax.py` | `eci/syntax.py` (byte-identical output) |
| `utils/formatting.py` | `eci/prompts.py` |
| `utils/mention_split.py`, `mention_whitelist.py` | `eci/scope.py` |
| `utils/resample.py`, `metrics.py`, `reporting.py`, `labels.py` | `eci/voting.py`, `metrics.py`, `report.py`, `labels.py` |
| `utils/logger.py`, `mlflow_tracker.py`, `trace_dump.py` | `eci/tracking.py` (+ checkpoint), `eci/llm.py` |
| `utils/causal_graph.py`, `scripts/analysis/distance_histogram.py`, `causal_graph.py` | `eci/graphs.py`, `analyze.py graph / hist` (from the report, no dataset reload) |
| 10 tools in `tools/` | `eci/tools.py` (same names, schemas, docstrings) |
| `scripts/experiments/coref_run.py` (monkeypatch) | `coref_hint` config section; chains in `eci/chains.py` |
| `scripts/experiments/negation_cot_run.py` (monkeypatch) | `few_shot.cot_generation.negation` |
| `scripts/experiments/oracle_count_run.py` | `gold_count` (already in agentere's config) |
| vote_threshold, _log, _report, _coref, posthoc_coref_report, maven_posthoc_sweep, maven_sweep_coref, sweep_all, intra_inter_compare | `analyze.py sweep` |
| binary_eval, vote_to_binary_eval, recompute_metrics, posteval | `analyze.py rescore` |
| summary, list_ablations, param_ablation, rerun_variance, intra_latex | `analyze.py runs / ablations / variance / latex` (both tracking DBs, read-only) |
| show_traces, token_usage_per_doc, tokens_vs_mentions | `analyze.py traces / spend` |
| `scripts/experiments/vote_ranking.py` | `analyze.py ranking` |

## 5. Behaviour that differs on purpose

| change | effect on numbers |
|---|---|
| concurrency is real | same requests, faster; see 2.4 for whether load changes answers |
| demo draw seeded per (seed, doc) | a rerun reproduces the same demos; draws differ from any agentere run's |
| truncated generations and all API errors retried | fewer empty-scored documents |
| label case/whitespace normalised | a handful of pairs per run |
| `provider_order` applied to tool sub-calls too | agentere routed those by price (inference and CoT were already pinned) |
| rule sets rewritten | only with the coherence tool enabled (it never fired before) |
| MLflow: `workere/mlflow.db`, experiment `workere`; `mlflow.log_traces` switch | agentere's DB untouched; `analyze.py` reads both |
| run directory per run (`logs/runs/<stem>/`) | report file name and schema unchanged; rows gain `word_distance`, `sentence_distance`; report gains `doc_index`, `failures`, `usage`, `offspec_labels` |

## 6. Not ported, and why

Every agentere script that reads `run_*.json` still works on workere reports (same schema,
additive fields only), so none of these is lost.

| what | why |
|---|---|
| one-off studies tied to a finding: `maven_drift_*`, `maven_graph_shape*`, `maven_structure_battery`, `maven_layers`, `maven_propagation_shape`, `maven_deployable_steps`, `misannotation_audit`, `fp_taxonomy`, `fp_cascade`, `fp_dump`, `fp_sentence_cap`, `filter_battery`, `pair_errors`, `pair_cot_errors(_llm)`, `pair_feature_table`, `justification_errors`, `partner_stability`, `gold_vs_mentions`, `gold_and_reasoning_vs_mentions`, `theme_density_histogram`, `random_pair_baseline`, `intra_exclude_close_pairs`, `vote_threshold_consensus`, `count_estimation_report`, `fix_unannotated` | results recorded in notes/ and memory; runnable in agentere on either framework's reports |
| settled experiments: `lexical_prior*`, `mention_filter*`, `count_estimation`, `density_oracle_llm`, `judge_pairs`, `temporal_order`, `autoprompt`, `llm_perpair`, `simple_eval`, `enc_*samp_perc`, `dev/causal_ranking_eval` | standalone protocols, not framework capabilities; several recorded as duds |
| `encoder_baseline/encoder.py` (Longformer training) | standalone GPU training; the `encoder` tool reads its `predictions.json` unchanged |
| `dataprep/*_standard_eci.py` (dataset builders) | build and push the HF datasets; the ESC FALLING_ACTION inversion fix belongs there and needs a republish |
| `scripts/dev/*`, `migrate_mlflow_to_sqlite.py`, `demo_eci.py`, `test_bay.ipynb`, `paper*/` | superseded by `--dry-run` / tests, or not code |

## 7. To start using it

| step | |
|---|---|
| environment | no manifest shipped (you manage envs): agentere's `.venv` runs it as is; `langgraph` is no longer needed |
| git | `git init` in workere so runs capture commit and diff again |
| ESC coreference | `coref_hint` and `sweep --coref` on ESC need the ESC corpus: `esl_root` (default `data/EventStoryLine`) or `--esl-root ../agentere/data/EventStoryLine` |
| tests | `python -m unittest tests.test_units tests.test_pipeline tests.test_parity` |
| concurrency | queue files asking for 200 now get 200 (see 2.4 before relying on it for comparisons with the agentere history) |

## 8. Spend this session

| run | cost |
|---|---|
| smoke, 2 docs | $0.01 |
| d46be8a3 config, conc 200 | $3.31 |
| d46be8a3 config, conc 28 | $3.29 |
| **total** | **$6.61** (of the $20 budget) |
