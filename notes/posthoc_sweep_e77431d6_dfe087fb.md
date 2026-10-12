# Post-hoc sweep — e77431d6 (ESC) and dfe087fb (MAVEN), 2026-09-28

Both runs: deepseek-v4-flash, coref-aware lean prompt (group mentions, emit every mention pair; no coref hint), 7 passes, 1 few-shot demo.

| run | dataset | docs | docs with ≥1 chain | chain source |
|---|---|---|---|---|
| e77431d6 | ESC v0.9 `-workere` (FALLING_ACTION fixed) | 233 | 221 | ECB+ (oracle) |
| dfe087fb | MAVEN-ERE test `-workere` | 710 | 426 | Maven-ERE-span (oracle) |

Scoring: ordered pairs (same convention as the logged `binary*` metrics). Sanity check: the majority (t≥4) base row reproduces the logged metric in both runs (ESC 46.6 = 46.6; MAVEN 31.1 = 31.1).
Chain join check: propagating the gold through the chains adds 0 pairs on MAVEN (exact) and 136 of 10,118 on ESC (1.3%).

Variants: `+coref` expands each predicted pair over both mentions' coreference chains. `+trans1` adds one directed transitive step (a→b, b→c ⇒ a→c). `+coref+trans1` propagates first and then takes one transitive step. `+trans1+coref` does the reverse; it gave the same result to ±0.1.

## Best F1 per variant (ALL scope; t = vote threshold)

| run | variant | best t | ALL P | ALL R | ALL F1 | INTRA F1 | INTER F1 | Δ ALL F1 vs base best |
|---|---|---|---|---|---|---|---|---|
| ESC e77431d6 | base | 3 | 39.7 | 57.3 | 46.9 | 55.9 | 43.3 | — |
| ESC e77431d6 | +coref | 5 | 47.6 | 54.0 | **50.6** | 55.1 | 48.9 | +3.7 |
| MAVEN dfe087fb | base | 3 | 24.9 | 41.5 | 31.1 | 48.4 | 26.1 | — |
| MAVEN dfe087fb | +coref | 4 | 26.8 | 36.6 | 31.0 | 47.8 | 26.0 | −0.1 |
| MAVEN dfe087fb | +trans1 | 4 | 24.4 | 43.0 | 31.2 | 49.5 | 26.3 | +0.1 |
| MAVEN dfe087fb | +coref+trans1 | 4 | 23.9 | 44.2 | 31.0 | 49.4 | 26.3 | −0.1 |
| MAVEN dfe087fb | +trans1+coref | 4 | 23.9 | 44.2 | 31.0 | 49.4 | 26.3 | −0.1 |
| MAVEN dfe087fb | +trans1(unanim) | 4 | 27.5 | 37.3 | 31.7 | 48.9 | 26.4 | +0.6 |
| MAVEN dfe087fb | +trans (fixpoint) | 5 | 25.4 | 39.1 | 30.8 | 48.5 | 26.2 | −0.3 |
| MAVEN dfe087fb | +coref+trans (fixpoint) | 5 | 24.8 | 40.4 | 30.7 | 48.5 | 26.2 | −0.4 |

Noise floor ≈ 1.2 F1 sd at full eval size, so every MAVEN delta is noise. On MAVEN, post-hoc propagation on top of the coref prompt adds nothing, as was already seen on dev (4c5d68e6). On ESC, propagation still gains +3.7, almost all of it inter-sentence (+5.6 INTER, −0.8 INTRA), and it moves the best threshold from 3 to 5.

## Full threshold sweeps (best t per variant in bold)

### ESC e77431d6

| t | variant | ALL P | ALL R | ALL F1 | INTRA F1 | INTER F1 |
|---|---|---|---|---|---|---|
| 1 union | base | 29.1 | 75.4 | 41.9 | 52.4 | 38.2 |
| 1 | +coref | 29.0 | 82.3 | 42.9 | 53.0 | 39.5 |
| 2 | base | 35.7 | 64.8 | 46.1 | 55.7 | 42.3 |
| 2 | +coref | 36.0 | 72.8 | 48.2 | 56.5 | 45.2 |
| 3 | base | 39.7 | 57.3 | **46.9** | 55.9 | 43.3 |
| 3 | +coref | 40.1 | 66.4 | 50.0 | 57.1 | 47.3 |
| 4 majority | base | 43.0 | 50.8 | 46.6 | 54.8 | 43.2 |
| 4 | +coref | 43.5 | 60.3 | 50.6 | 56.1 | 48.5 |
| 5 | base | 47.1 | 44.4 | 45.7 | 53.8 | 42.3 |
| 5 | +coref | 47.6 | 54.0 | **50.6** | 55.1 | 48.9 |
| 6 | base | 51.0 | 36.6 | 42.6 | 49.2 | 39.7 |
| 6 | +coref | 51.6 | 45.6 | 48.4 | 51.5 | 47.1 |
| 7 unanimous | base | 56.8 | 26.7 | 36.3 | 40.5 | 34.4 |
| 7 | +coref | 55.8 | 32.8 | 41.3 | 42.7 | 40.7 |

### MAVEN dfe087fb

| t | variant | ALL P | ALL R | ALL F1 | INTRA F1 | INTER F1 |
|---|---|---|---|---|---|---|
| 1 union | base | 17.0 | 59.2 | 26.4 | 45.9 | 22.2 |
| 1 | +coref | 16.6 | 59.6 | 25.9 | 45.6 | 21.8 |
| 1 | +trans1 | 13.4 | 70.7 | 22.5 | 45.1 | 18.9 |
| 1 | +coref+trans1 | 13.1 | 71.1 | 22.1 | 44.8 | 18.6 |
| 2 | base | 21.6 | 47.8 | 29.7 | 47.9 | 24.9 |
| 2 | +coref | 21.1 | 48.9 | 29.5 | 47.7 | 24.8 |
| 2 | +trans1 | 17.7 | 58.0 | 27.1 | 47.9 | 22.8 |
| 2 | +coref+trans1 | 17.3 | 58.9 | 26.8 | 47.7 | 22.6 |
| 3 | base | 24.9 | 41.5 | **31.1** | 48.4 | 26.1 |
| 3 | +coref | 24.1 | 42.4 | 30.7 | 48.3 | 25.8 |
| 3 | +trans1 | 21.4 | 50.3 | 30.0 | 49.3 | 25.3 |
| 3 | +coref+trans1 | 20.9 | 51.4 | 29.7 | 49.2 | 25.1 |
| 4 majority | base | 27.7 | 35.5 | 31.1 | 47.9 | 25.9 |
| 4 | +coref | 26.8 | 36.6 | **31.0** | 47.8 | 26.0 |
| 4 | +trans1 | 24.4 | 43.0 | **31.2** | 49.5 | 26.3 |
| 4 | +coref+trans1 | 23.9 | 44.2 | **31.0** | 49.4 | 26.3 |
| 5 | base | 30.2 | 30.0 | 30.1 | 46.3 | 24.9 |
| 5 | +coref | 29.1 | 31.3 | 30.2 | 46.3 | 25.2 |
| 5 | +trans1 | 27.3 | 36.0 | 31.1 | 48.3 | 26.1 |
| 5 | +coref+trans1 | 26.5 | 37.3 | 31.0 | 48.3 | 26.2 |
| 6 | base | 33.2 | 24.2 | 28.0 | 43.0 | 22.9 |
| 6 | +coref | 31.5 | 25.5 | 28.2 | 43.0 | 23.4 |
| 6 | +trans1 | 30.6 | 28.5 | 29.5 | 45.5 | 24.5 |
| 6 | +coref+trans1 | 29.3 | 29.9 | 29.6 | 45.5 | 24.9 |
| 7 unanimous | base | 38.0 | 17.6 | 24.1 | 36.8 | 19.6 |
| 7 | +coref | 38.3 | 18.7 | 25.1 | 37.1 | 21.1 |
| 7 | +trans1 | 35.9 | 19.8 | 25.5 | 38.9 | 21.0 |
| 7 | +coref+trans1 | 36.2 | 21.0 | 26.6 | 39.1 | 22.4 |

Every variant, with pair counts: `uv run python analyze.py sweep e77431d6 --coref` and `uv run python analyze.py sweep dfe087fb --coref --closure` (about 10–20 s each, no model calls).
