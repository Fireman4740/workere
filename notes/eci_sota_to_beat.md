# ECI: what to beat (2026-10-08)

Best published numbers on ESC, CTB and MAVEN-ERE, from `litt/` plus CarICL (Feng et al., IPM 63, 2026, not in `litt/papers`).
Every number is causal existence (direction ignored), which is what workere's `binary` measures.
The settings differ in which pairs are scored, so compare a run with the column that matches it.

## Our runs

workere, deepseek-v4-flash, 1 random shot with CoT, 7-run majority vote. P / R / F1.

| run | dataset | docs | all pairs | intra | inter | cost |
|---|---|---|---|---|---|---|
| a43655dd | CTB (CAT, same-sentence pairs only) | 183 | 65.7 / 61.1 / **63.3** | 65.9 / 61.1 / 63.4 | – | $1.72 |
| e77431d6 | ESC v0.9 | 233 | 43.1 / 50.8 / **46.6** | 54.7 / 55.1 / 54.9 | 38.9 / 48.8 / 43.3 | $4.50 |
| dfe087fb | MAVEN-ERE test | 710 | 27.7 / 35.4 / **31.1** | 44.5 / 51.6 / 47.8 | 22.8 / 30.1 / 26.0 | $13.46 |

## Sentence level: every pair inside one sentence, trained or with training demos

ESC preprocessing follows Gao et al. 2019: 4 event types dropped, 7,805 intra pairs. The last 2 topics are dev, with 5-fold CV over the rest. CTB uses 10-fold CV. All numbers are F1.
Excluded: CPATT and fine-tuned SERE. Their test negatives are randomly paired events rather than every pair, so they are not comparable; SERE itself keeps them out of its main results.

| method | venue | model | ESC | CTB |
|---|---|---|---|---|
| CLECI | Information 2026 | RoBERTa-large | **75.7** | **83.3** |
| DiffusECI | 2024 | PLM | 71.4 | 75.4 |
| DICP | EMNLP 2025 | RoBERTa | 71.3 | 67.3 |
| LKCER | COLING 2025 | PLM + LLM knowledge | 69.8 | 68.5 |
| CarICL | IPM 2026 | Qwen2.5-7B, LoRA + ORPO, 8 retrieved demos | 67.9 | 69.5 |
| CarICL, prompt only | IPM 2026 | GPT-3.5-Turbo, 8 retrieved demos, no training | 66.4 | 68.0 |
| SemDI | EMNLP 2024 | PLM | 62.0 | 67.0 |
| GPT-4, zero-shot | Li et al. 2024 | – | 45.2 | 8.7 |

Our closest points: ESC intra 54.9, scored on all event types rather than Gao's subset. CTB 63.3, scored on all CAT pairs of the whole corpus.

## Whole document: intra + inter pairs

| method | venue | ESC all (intra / inter) | MAVEN-ERE all (intra / inter) | CTB |
|---|---|---|---|---|
| iLIF | ACL 2024 | **62.5** (71.2 / 59.1) | 57.1 (60.9 / 56.8) | – |
| GLM4ECI, LLaMA | ACL 2025 | 57.1 | **65.6** | – |
| GLM4ECI, BART | ACL 2025 | 60.7 | 64.4 | – |
| PPAT | IJCAI 2023 | 54.5 (65.2 / 49.3) | 39.2 (49.0 / 35.5) | 66.2 |

## LLM prompting without training: Gao et al. 2023 setting, one pair per query

| method | model | ESC | CTB | MAVEN-ERE |
|---|---|---|---|---|
| SERE | GPT-4o-mini | **49.9** | **20.0** | **42.3** |
| Dr.ECI | PaLM2 | 41.9 | 13.0 | 34.5 |
| zero-shot | GPT-3.5-Turbo | 41.0 | 12.8 | – |

## CarICL

Feng, Li, Qin and Li, *Information Processing and Management* 63 (2026) 104500. It is sentence-level only, with no MAVEN-ERE results.

Each query is a multiple-choice prompt. The prompt includes 8 demonstrations, retrieved by embedding "Represent the causality between e1 and e2" together with the sentence (BGE-large).
Each demonstration carries a rationale that GPT-3.5 wrote from the gold label. The model is then trained with ORPO on these same prompts over the training folds.

Ablation, F1 lost on ESC / ESC with a random sentence split / CTB:

| removed | ESC | ESC random | CTB |
|---|---|---|---|
| causality-aware retrieval (random demos instead) | −8.7 | −7.7 | −10.1 |
| gold-conditioned rationales | −5.9 | −6.3 | −5.5 |
| ORPO training | −11.9 | −9.1 | −10.6 |

Precision rises steadily with the number of shots, from about 20 at 0 shots to 66 at 8 on ESC, while recall stays near 60–70.

Caveats:
- The paper does not compare against DICP, CLECI or DiffusECI, all of which score higher on ESC.
- The CTB folds are stratified by label, and the paper does not say whether it splits by pair or by sentence. If it splits by pair, a test sentence's other pairs can be retrieved as demonstrations. That would explain part of the prompt-only jump from 12.8 to 68.0 on CTB.
