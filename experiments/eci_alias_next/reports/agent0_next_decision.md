# Agent 0 - decision and launch status, 2026-10-27

## Material actually received

The attachment is the research report for the **virtual aliases** track (agent 6), not the six agents' combined empirical results. It contains one local diagnostic on metadata and a projector, and a prospective plan. It does NOT contain new ECI F1 measurements from actual model calls. The historical project overview is separately available in prior conversation context.

## Evidence-based decisions

1. **Do not assert that the 88-document reserve is untouched.** The report claims exact 88/88 ID overlap with H1 jobs and summaries; independent verification requires the underlying manifests, not present here.
2. **Keep old Luna x d1 and threshold 0.39 as a historical reference only.** Alias vote `q >= 4/7` is a different score and baseline must be rerun under that new schema. Do not silently splice q into frozen formula.
3. **Start with A vs R, not all expansions.** A uses new handles; R duplicates the same records but retains the original IDs. Both must share selected mentions, demos, model version, decoding and provider budgets. The package prepares 7 slots per condition, 14 per document.
4. **Do not use the labels in selection or prompts.** Gold is only for analysis after predictions are fixed.
5. **Do not launch 616 paid calls without repository/provider access and usage gating.** The user-provided report estimates up to 616 A/R calls for 44 documents; actual cost is not known in this session.

## Actions actually completed here

- Independently implemented a standard-library experiment kit with strict input normalization, A/R/B/P view generation, alias projection, per-slot vote count and gold-separated F1 evaluation.
- Executed and passed 37 unit tests locally.
- Successfully ran an end-to-end **synthetic-only** experiment data flow (3 documents, 4 arms, 7 slots each; 84 planned but NOT sent calls).
- Implemented provider usage and fair-budget gate; it cannot pass on real usage without actual model logs.
- Attempted connected GitHub repository discovery for `workere`; no accessible matching repository was returned.

## Blockers to actual Luna comparison

- Private repository/caches and exact DEV44 doc IDs not accessible in this session.
- Exact historical prompt, 7 out-of-topic demonstrations, Luna provider route and tokenizer not supplied.
- No manifest of actual model responses, usage or gold labels provided.

## Recommended next actions (priority)

**Gate 1 (no model spend):** Integrate the normalizer into `eci/data.py`; construct real `public_docs_DEV44.jsonl` and verify exact candidate/mention IDs, repeated, nested or discontinuous token spans; check the adapter against current `eci/voting.py`.

**Gate 2 (no spend):** Render A/R against the exact frozen prompt/demos; provider tokenizer must verify all tokens/truncation; check A/R selected mentions and bytes unchanged. Freeze prompt/version/hash in a run manifest.

**Gate 3 (small smoke spend after operator authorizes):** Run one development document x 7 A and 7 R slots; verify JSON schema and projection, then label-blind quality inspection. These exploratory calls must be logged and not confused with confirmation.

**Gate 4 (larger spend, explicitly approved):** Execute the 44-document A/R pilot; aggregate, evaluate, bootstrap by topic and decide stop/continue. Only extend to B/P after a useful A/R contrast and a proper budget plan.

**Gate 5 (confirmation):** Identify a genuinely unexposed set. H1 overlap prevents treating the previously advertised 88 IDs as a clean final holdout.

## Hypotheses and stopping criteria

- H1: redundant mention views recover missed original pairs at controlled cost.
- H2: distinct alias IDs improve over **exactly matched record repetition**.
- Stop the alias-specific mechanism if projection removes its gain, A fails to beat R on useful marginal precision and net F1, or token/sample artifacts explain the difference.
- Distinguish no statistical evidence from evidence of practical equivalence; with few topics uncertainty can be large.

All synthetic performance numbers in `examples/out/` are fabricated demonstrations; do not report them as empirical results.
