# Jev / Laya decision models on intra-sentence ECI (2026-09-29)

**Verdict: Laya zero-shot is at chance on ECI, and a temperature refit confirms it (the fitted T goes to infinity).
Stopped at the dev pilot; no full run. Jev not tested (no API key).**

## The models

| model | who | access | architecture |
|---|---|---|---|
| Jev 1.13 | TypeSafe | closed hosted API (`POST thejevai.com/v1/systemone`), paid credits from $10 / 100k | undisclosed |
| Laya (`convaiinnovations/laya`) | Convai Innovations | open weights, Apache-2.0 | ModernBERT-large 421M + decision head; 512-token context |

Both take a *state* plus typed questions (`choice` / `score` / `noul`) and return a probability per option in one
forward pass. Laya's own card says the base checkpoints are "near chance on typed-decisions zero-shot" and that
the capability comes from fine-tuning.

## Setup

- Data: the three latest `Nofing/*-standard-eci-workere` repos. Every intra-sentence unordered mention pair is one decision.
- State: the sentence with both mentions wrapped in `« »`, plus the two surfaces.
- Laya 0.3.21 run from a scratchpad copy on `PYTHONPATH` with `agentere/.venv` (nothing installed). CPU with bf16
  autocast: agentere's torch 2.6/cu124 has no sm_120 kernels, so the RTX 5060 is unusable.
  Throughput was 2–5 pairs/s.
- Score = 1 − P(no relation). Option keys are neutral letters, as the card recommends over `noul`.

| phrasing | question | options |
|---|---|---|
| dir3 | "The text marks two events with « ». Is one of them a cause of the other?" | A 1→2 · B 2→1 · C neither |
| bin2 | same | A yes · B no |
| nli2 | "Is the claim supported by the text?" (state + "Claim: one of the two marked events … is a cause of the other.") | A yes · B no |

Gate set in advance: go to the full run (~49k pairs, ~3–4 h CPU) only if a phrasing reaches dev AUROC ≥ 0.6.

## Pilot results (dev splits only, `laya` English)

| phrasing | split | pairs | gold | base rate | AUROC | AP | says causal | argmax F1 | best-threshold F1 (oracle) | all-positive F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| dir3 | ESC dev | 1,343 | 171 | 0.127 | 0.539 | 0.145 | 95.6% | 0.232 | 0.245 | 0.226 |
| bin2 | ESC dev | 1,343 | 171 | 0.127 | 0.561 | 0.156 | 86.4% | 0.248 | 0.250 | 0.226 |
| nli2 | ESC dev | 1,343 | 171 | 0.127 | 0.509 | 0.134 | 90.7% | 0.230 | 0.233 | 0.226 |
| dir3 | MAVEN dev (40 docs, 576/1,065 pairs, run cut) | 576 | 90 | 0.156 | 0.567 | 0.177 | 98.6% | 0.274 | 0.305 | 0.270 |

Direction (dir3, gold causal pairs only):

| split | gold pairs | direction accuracy | "earlier mention is the cause" baseline | says 1→2 |
|---|---|---|---|---|
| ESC dev | 171 | 0.702 | 0.626 | 92.4% |
| MAVEN dev (partial) | 90 | 0.844 | 0.844 | 100% |

The model picks option A for 86–92% of pairs whatever A means ("causes", "yes", or "supported"), and its best
threshold barely beats labelling every pair causal. Direction is text order.

## Temperature refit

The pilot ran with the shipped temperatures (`choice:2` 1.906, `choice:3-5` 1.760), not refit on this data; the
card and [Mysore's write-up](https://medium.com/@visrow/what-is-laya-laya-vs-jev-with-live-demo-42c2ab494e02)
say to refit one T per (question type, option count) on target data. Refit on the saved outputs
(`baselines/decision_eci_recal.py`, logits = T_shipped · log p), 2-fold cross-fitted over documents:

| run | method | fitted T | NLL | ECE | mean P(causal) | AUROC | AP | argmax F1 | oracle F1 |
|---|---|---|---|---|---|---|---|---|---|
| dir3 ESC dev | shipped | 1.76 | 2.617 | 0.659 | 0.913 | 0.539 | 0.145 | 0.232 | 0.245 |
| | temperature | ~1.6e7 (→ ∞) | 1.099 (= ln 3) | 0.203 | 0.667 | 0.534 | 0.135 | 0.232 | 0.251 |
| | bias + temperature | 3.1 / 1.7 | 0.447 | 0.038 | 0.130 | 0.537 | 0.139 | 0.000 | 0.243 |
| bin2 ESC dev | shipped | 1.91 | 1.829 | 0.602 | 0.789 | 0.561 | 0.156 | 0.248 | 0.250 |
| | temperature | ~1.7e7 (→ ∞) | 0.693 (= ln 2) | 0.246 | 0.500 | 0.562 | 0.156 | 0.248 | 0.248 |
| | bias + temperature | 57 / 6.0 | 0.384 | 0.010 | 0.133 | 0.542 | 0.150 | 0.000 | 0.249 |
| nli2 ESC dev | shipped | 1.91 | 1.307 | 0.535 | 0.720 | 0.509 | 0.134 | 0.230 | 0.233 |
| | temperature | ~1.0e7 (→ ∞) | 0.693 (= ln 2) | 0.296 | 0.500 | 0.509 | 0.134 | 0.230 | 0.233 |
| | bias + temperature | 960 / 15 | 0.383 | 0.019 | 0.128 | 0.473 | 0.122 | 0.000 | 0.228 |
| dir3 MAVEN dev (partial) | shipped | 1.76 | 2.559 | 0.617 | 0.923 | 0.567 | 0.177 | 0.274 | 0.305 |
| | temperature | ~1.7e7 (→ ∞) | 1.099 (= ln 3) | 0.188 | 0.667 | 0.550 | 0.167 | 0.274 | 0.302 |
| | bias + temperature | 4.2 / 5.7 | 0.513 | 0.034 | 0.158 | 0.552 | 0.178 | 0.000 | 0.302 |

- The NLL-optimal temperature is infinite in every case: the calibrated model is the uniform distribution,
  i.e. the logits carry no usable information about the answer.
- Adding a bias per option (which also removes the option-A lean) collapses to the base rate
  (mean P(causal) ≈ 0.13 vs 0.127–0.156): ECE drops to 0.01–0.04 and it predicts nothing causal.
- A temperature cannot change the argmax, and on two-option questions cannot change the ranking, so AUROC, AP
  and oracle F1 stay at chance. The small AUROC moves come from the two folds getting different fits.

## Why it fails: it scores the sentence, not the pair

The load is sound: the card's own example gives billing 0.97 / churn 0.82 (documented 0.94 / 0.89), and an outage
ticket flips to technical 0.95 / churn 0.07. The failure is in what the model reads:

| run | score variance explained by sentence | pair AUROC | pair AUROC within sentences holding both classes | sentence AUROC (max score vs "has a causal pair") |
|---|---|---|---|---|
| dir3 ESC dev | 0.45 | 0.539 | 0.495 (74 sentences) | 0.592 (155) |
| bin2 ESC dev | 0.52 | 0.561 | 0.498 (74) | 0.689 (155) |
| nli2 ESC dev | 0.53 | 0.509 | 0.476 (74) | 0.707 (155) |
| dir3 MAVEN dev (partial) | 0.38 | 0.567 | 0.511 (34) | 0.610 (142) |

Laya detects causal *sentences* (AUROC ~0.7) but is at chance on *which pair* in a sentence (0.48–0.51): the
« » markers and the named events are ignored. Its decision head was trained on whole-text properties (routing,
urgency, churn) and scores each option from [MASK] tokens attending over the whole sequence; nothing taught it to
bind a question to two marked spans. The ModernBERT-large ECI results in the literature come from supervised
fine-tuning with a head on the two event tokens' embeddings. That is a different model from this zero-shot
decision head on the same encoder.

## For scale (same datasets, intra-sentence binary F1)

| system | ESC (233 docs) | CTB (183) | MAVEN test (710) |
|---|---|---|---|
| all pairs causal | 0.301 | 0.059 | 0.290 |
| workere, deepseek-v4-flash (best / range of finished runs) | 0.549 / 0.499–0.549 | 0.634 / 0.592–0.634 | 0.478 / 0.448–0.478 |
| Laya zero-shot | not run (dev at chance) | not run | not run |

## Not done

| item | why |
|---|---|
| Jev | Needs `JEV_API_KEY` and paid credits. `decision_eci.py --backend jev` implements the documented request shape but is untested; start with `--limit 200` to measure credits per pair. |
| `laya-multilingual`, `laya-typed-decisions` | The card reports both at chance zero-shot (0.34 on typed-decisions); typed-decisions is a specialist for four unrelated workflows. |
| swapped-option controls, remaining MAVEN dev, full eval | Stopped at the gate. |
| fine-tuning Laya on MAVEN train (77,887 intra pairs) | The only route the card supports. Needs a GPU-capable torch (cu128) in some venv; about 4–5 h on 2×T4 per the card. |

## Files

- `workere/baselines/decision_eci.py`: `predict` (laya / jev, resumable jsonl) and `score`
- `workere/baselines/decision_eci_recal.py`: temperature and bias+temperature refit, cross-fitted
- `workere/baselines/run_decision_eci.sh`: the full-run driver (unused)
- `workere/logs/decision_eci/pilot/*.jsonl`: per-pair probabilities

Sources: [Laya model card](https://huggingface.co/convaiinnovations/laya) ·
[laya-typed-decisions](https://huggingface.co/convaiinnovations/laya-typed-decisions) ·
[featherless-ai/simple-jev](https://github.com/featherless-ai/simple-jev) ·
[What is Jev AI](https://huggingface.co/blog/sora-2/what-is-jev-ai-a-practical-guide-to-system-one-and)
