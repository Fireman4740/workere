# Run-to-run spread of F1, rebuilt from the Novita traces (2026-10-05)

`notes/pass_variance.py` → `notes/pass_variance.json`. The metric is binary F1 on ordered pairs, as
logged (`binary*`). A pair is predicted when at least t of the N passes label it causes/causedby.
Every sd is in F1 points.

## Runs pooled

| | 7-pass run | 1-pass runs | second 7-pass run | replicates per doc (slot 0 / slots 1–6) | docs scored |
|---|---|---|---|---|--:|
| CTB | a43655dd | 525746e4, 1dd870a8 (crashed at 159 docs) | — | 2–3 / 1 | 138 of 183 |
| ESC | e77431d6 | 03dd2a75 | — | 2 / 1 | 232 of 233 |
| MAVEN-ERE | dfe087fb | a0293913 | 1ac866ca, Novita calls only (3,154 kept, 1,816 Baidu-priced dropped) | 2–3 / 1–2 | 707 of 710 |

- **No 3-pass Novita run exists.** The 10-05 runs show `n_runs: 3` in MLflow, but they ran with
  `resampling.enabled: false`, so they made one pass per document. The only 3-pass runs
  (ce228f72, 96e7ec99, 409969bf) were served by Baidu.
- **Unscored documents.** These have no gold and no prediction in any pass, so they add nothing to
  F1 (45 on CTB).
- **Reconstruction.** Each pass was parsed from its trace with `eci.parse` and re-voted with
  `eci.voting`. The vote tallies match the checkpoint on every document of the 8 runs. The 7-pass
  F1 at t=4 equals the logged binary F1 (63.3, 46.6, 31.1).
- **Identical requests.** For each slot r, the 7-pass runs and the 1-pass runs (r=0) send
  byte-identical requests (system, demo, demo CoT, test prompt), with 0 mismatches. So each
  document has 7 demo **slots** (distinct demos), and each slot has 1–3 decoding **replicates**.
- **Same config.** The runs differ only in the resampling switch and `provider_order`. Every
  pass kept was served at Novita's price ($0.28/M completion).

## Method

| | what varies | estimator |
|---|---|---|
| new seed | demo draw (same pool of 20) + decoding | For each document, take two disjoint sets of N slots and compute half the squared difference of the document's F1 contribution (delta method on TP and FP; gold fixed). Average over all such pairs and over replicates, then sum over documents. Unbiased for N ≤ 3. |
| same seed | decoding only (the identical config rerun) | The same estimator on two runs that use different replicates of the same slots: slot 0 on all datasets; slots 0–6 on the 447 MAVEN docs where all 7 passes of 1ac866ca are Novita |
| N = 7 | — | Two disjoint 7-pass runs would need 14 slots. The sd comes from a bootstrap that draws each document's 7 passes from its 7 slots with replacement. F1 is that of the run itself. |

The variance is over passes on a fixed document set: the noise between two runs on the same
documents, not document-sampling error. Brackets give a 95% interval for the sd estimate, from
resampling documents.

## Results

**sd of F1, new seed, all pairs** (N=1 and N=3: expected F1 over demo draws; N=7: the run)

| | docs scored | N=1: F1 | sd | N=3, t=2: F1 | sd | N=7, t=4: F1 | sd (bootstrap) |
|---|--:|--:|--:|--:|--:|--:|--:|
| CTB | 138 | 58.2 | 1.79 [1.47–2.13] | 61.2 | 1.45 [1.17–1.70] | 63.3 | 1.14 |
| ESC | 232 | 44.1 | 0.73 [0.64–0.83] | 45.9 | 0.59 [0.49–0.69] | 46.5 | 0.49 |
| MAVEN-ERE | 707 | 28.7 | 0.63 [0.48–0.79] | 30.4 | 0.54 [0.37–0.70] | 31.3 | 0.43 |

**Smallest difference between two independent runs that is significant at 95%** (1.96·√2·sd)

| | N=1 | N=3 (t=2) | N=7 (t=4) |
|---|--:|--:|--:|
| CTB | 5.0 | 4.0 | 3.2 |
| ESC | 2.0 | 1.6 | 1.3 |
| MAVEN-ERE | 1.8 | 1.5 | 1.2 |

**By scope** (CTB has no inter-sentence gold)

| | scope | N=1: F1 | sd | N=3, t=2: F1 | sd | N=7, t=4: F1 | sd (bootstrap) |
|---|---|--:|--:|--:|--:|--:|--:|
| CTB | intra | 58.5 | 1.75 [1.45–2.06] | 61.3 | 1.45 [1.17–1.71] | 63.4 | 1.14 |
| ESC | intra | 52.5 | 0.80 [0.72–0.88] | 54.4 | 0.64 [0.56–0.73] | 55.1 | 0.53 |
| ESC | inter | 40.7 | 0.83 [0.71–0.95] | 42.4 | 0.67 [0.55–0.79] | 43.0 | 0.55 |
| MAVEN-ERE | intra | 45.5 | 0.51 [0.47–0.54] | 47.0 | 0.42 [0.39–0.45] | 47.6 | 0.33 |
| MAVEN-ERE | inter | 24.0 | 0.78 [0.56–0.99] | 25.4 | 0.68 [0.44–0.91] | 26.2 | 0.54 |

**Every threshold, N=3** (all pairs)

| | t=1 F1 | sd | t=2 F1 | sd | t=3 F1 | sd |
|---|--:|--:|--:|--:|--:|--:|
| CTB | 57.4 | 1.14 | 61.2 | 1.45 | 55.9 | 1.81 |
| ESC | 44.2 | 0.52 | 45.9 | 0.59 | 41.5 | 0.85 |
| MAVEN-ERE | 28.2 | 0.42 | 30.4 | 0.54 | 27.4 | 0.56 |

**Every threshold, N=7** (F1 of the run / bootstrap sd, all pairs)

| | t=1 | t=2 | t=3 | t=4 | t=5 | t=6 | t=7 |
|---|--:|--:|--:|--:|--:|--:|--:|
| CTB | 54.0 / 0.72 | 59.1 / 0.97 | 62.3 / 1.04 | 63.3 / 1.14 | 61.2 / 1.29 | 57.3 / 1.42 | 49.7 / 1.24 |
| ESC | 42.2 / 0.31 | 46.0 / 0.42 | 46.9 / 0.46 | 46.5 / 0.49 | 45.7 / 0.53 | 42.5 / 0.64 | 36.3 / 0.63 |
| MAVEN-ERE | 26.3 / 0.27 | 30.0 / 0.35 | 31.3 / 0.38 | 31.3 / 0.43 | 30.0 / 0.45 | 28.0 / 0.45 | 23.7 / 0.40 |

**Smallest significant gap between the means of 3 runs per config** (95%, two-sided; each run
with its own `seed`)

| | N=1 | N=3 (t=2) | N=7 (t=4) |
|---|--:|--:|--:|
| sd known from this note: 1.96·√(2/3)·sd | | | |
| CTB | 2.9 | 2.3 | 1.8 |
| ESC | 1.2 | 0.9 | 0.8 |
| MAVEN-ERE | 1.0 | 0.9 | 0.7 |
| sd estimated from the 6 runs only (t, 4 df): 2.78·√(2/3)·sd | | | |
| CTB | 4.1 | 3.3 | 2.6 |
| ESC | 1.7 | 1.3 | 1.1 |
| MAVEN-ERE | 1.4 | 1.2 | 1.0 |

**Same with 5 runs per config** (1.96·√(2/5)·sd; in brackets, sd estimated from the 10 runs
only, t with 8 df)

| | N=1 | N=3 (t=2) | N=7 (t=4) |
|---|--:|--:|--:|
| CTB | 2.2 [2.6] | 1.8 [2.1] | 1.4 [1.7] |
| ESC | 0.9 [1.1] | 0.7 [0.9] | 0.6 [0.7] |
| MAVEN-ERE | 0.8 [0.9] | 0.7 [0.8] | 0.5 [0.6] |

Under a single seed, MAVEN at N=3 floors at 1.0 with 5 runs (1.1 with 3), since the demo share
does not average out.

`seed` sets the per-document demo draw, so three reruns under one seed see the same demos and
average away only the decoding noise. On MAVEN at N=3, that leaves 32% of the variance in place:
the gap becomes 1.1 instead of 0.9. On CTB and ESC the demo share cannot be told from 0. The
N=7 column inherits the bootstrap's low bias.

### Same seed vs new seed: decoding is most of the noise

| | docs | N, t | same-seed sd | new-seed sd | demo share of variance |
|---|--:|---|--:|--:|--:|
| CTB | 138 | 1 | 1.95 [1.38–2.55] | 1.79 [1.47–2.13] | ~0 (estimate < 0) |
| ESC | 232 | 1 | 0.82 [0.66–0.98] | 0.73 [0.64–0.83] | ~0 (estimate < 0) |
| MAVEN-ERE | 707 | 1 | 0.47 [0.41–0.53] | 0.63 [0.48–0.79] | 45% |
| MAVEN-ERE, twin docs | 447 | 1, 1 | 0.57 [0.50–0.64] | 0.73 [0.60–0.88] | 39% |
| MAVEN-ERE, twin docs | 447 | 3, 2 | 0.50 [0.42–0.58] | 0.60 [0.46–0.77] | 32% |
| MAVEN-ERE, twin docs | 447 | 7, 4 | 0.52 [0.33–0.72] | — (bootstrap 0.48) | — |

Agreement between two passes of a document (mean Jaccard of the predicted pair sets):

| | identical request (temperature 0) | different demo |
|---|--:|--:|
| CTB | 0.655–0.667 | 0.619 |
| ESC | 0.607 | 0.577 |
| MAVEN-ERE | 0.463–0.490 | 0.465 |

Same-day and week-apart replicates agree equally well (CTB 10-05 vs 10-05: 0.667; 09-28 vs 10-05:
0.655–0.663), so the endpoint did not drift between 09-28 and 10-05.

### Checks

The seven single-pass runs inside each 7-pass run, against the N=1 estimate:

| | slot 0 | 1 | 2 | 3 | 4 | 5 | 6 | sd of the 7 | estimated sd |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| CTB | 62.5 | 60.2 | 58.4 | 56.0 | 57.6 | 59.6 | 54.6 | 2.67 | 1.79 |
| ESC | 43.9 | 44.0 | 43.1 | 44.6 | 44.8 | 43.1 | 44.8 | 0.75 | 0.73 |
| MAVEN-ERE | 28.1 | 29.0 | 28.8 | 29.1 | 29.0 | 28.5 | 28.5 | 0.37 | 0.63 |

Bootstrap sd divided by the unbiased sd, where both exist:

| | N=1 | N=3, t=1 | N=3, t=2 | N=3, t=3 |
|---|--:|--:|--:|--:|
| CTB | 0.93 | 0.91 | 1.01 | 0.88 |
| ESC | 0.92 | 0.91 | 1.01 | 0.83 |
| MAVEN-ERE | 0.95 | 0.97 | 1.00 | 0.98 |

- **Exact vs linearized.** Without the delta method (the corpus F1 of two disjoint N-slot runs,
  400 random splits per document), every N ≤ 3 sd agrees with the table within 0.05.
- **MAVEN twin runs.** On the 447 twin docs, the two 7-pass runs score 31.6 and 31.8 at t=4.

### Reading

- **Decoding is most of the noise.** At temperature 0, identical requests agree barely more than
  passes with different demos. Rerunning with the same seed is nearly as noisy as drawing new
  demos: about 60% of the variance on MAVEN, and not distinguishable from all of it on CTB and
  ESC.
- **More passes help less than 1/√N.** Going from N=1 to N=7 cuts the sd by 36% (CTB), 33% (ESC)
  and 32% (MAVEN), not by the 62% that 1/√N would give. Pairs the model emits about half the
  time stay coin flips under any majority.
- **N=7 sds are lower bounds.** At N=3, t=2 the bootstrap equals the unbiased estimate, but at
  union and unanimity it runs 5–17% low. On MAVEN at N=7 it sits 8% below the direct same-seed
  figure, and that figure leaves out the demo share. Read the N=7 sds as lower bounds, likely
  10–25% low.
- **CTB may be underestimated.** CTB's seven single-pass runs spread 2.67, above the interval of
  the estimate (1.47–2.13; p ≈ 0.04 with 6 df). Slot 0 sits about 3 points above the other slots
  in all three of its replicates (62.5, 60.1, 61.9). The CTB sds may be low, and the same-seed
  CTB figure is too coarse (138 docs, 2–3 replicates) to rule out a demo effect. MAVEN's slot runs
  err the other way (0.37 vs 0.63, p ≈ 0.09).
