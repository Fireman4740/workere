# MAVEN200: explicit coreference and structured inter-sentence search

Preregistered 2026-11-13 before either experiment is launched. Both experiments
are siblings of historical node `33718e74-4f38-47df-9a9f-aa68c4afba3a`.
They build on its frozen long-reading recipe, not an as-yet-unknown budget winner.
Each experiment compares treatment and control on its own single L40S.
The fixed entry point remains `bash steering/run_pilot.sh`.

## Common contract

Exactly the same SHA256-pinned historical 200 documents and five train demos.
Frozen Qwen3.8-27B AutoRound checkpoint and HyperQwen SIF; context 65536,
KV fp8, batch profile, concurrency 32, greedy decoding, natural thinking stop.
Preserve the historical batch profile's int8 MLP activations explicitly.
Actual weight/config/tokenizer/SIF bytes are hashed before GPU allocation.
Training demonstrations retain their known causal annotations. Evaluation gold
never enters a model request or candidate-selection function.
Public jobs are frozen separately from scoring data. Controls and treatments
are interleaved with alternating order across demonstration passes.

Primary endpoint for each experiment: treatment minus control micro-F1 of
unordered causal presence at vote >=3/5, paired over the same 200 documents.
5000 document bootstrap draws, seed 20261112. Nominal, unadjusted 95% intervals;
two exploratory primary hypotheses, no independent-confirmation claim.
Secondary: comparison with historical long vote@5, thresholds 1/2/4/5,
precision/recall, intra/inter and distances 1, 2–3, >=4, additions/removals TP/FP,
invalid and truncated responses, actual token and input costs. Original 200
documents are previously exposed development data.
Invalid final answers count as empty reads, with five passes retained.
Transport errors remain missing and prohibit final scoring. Per-stage atomic
records allow interrupted runs to resume without regenerating completed work.

## E1 — explicit event groups and causal graph

Experiment: `1e3c4d99-d066-4d14-beb4-3d4db9d94353`, arm `COREF`.

For each document/demo, generate two independent graphs with the same schema,
20480 thinking maximum and 4096 JSON maximum:

* `singleton`: every mention is a separate node; no identity merges.
* `coref`: predict disjoint event-identity groups and causal edges between their
  representative mention IDs. Expand an edge by Cartesian product in code.

Both use identical causal guidance and the same five transformed training
examples. No gold coreference demonstrations are added. Omitted mentions are
singletons. Overlapping groups, unknown IDs and within-group causal edges make
the reading invalid. No automatic transitive causal closure is applied.
Total: 2000 graph readings, balanced across methods.

Evaluate `coref_no_expansion` from the exact same generated graphs as a secondary
mechanism check. Separately propagate historical predictions through gold
clusters offline as `oracle_propagation_diagnostic`: this is a diagnostic, not
a deployable system or a guaranteed upper bound on F1. Oracle labels never
reach public jobs or inference. The frozen source parquet's checksum is checked.

Motivation includes the project's earlier negative result: predicted-coreference
propagation reduced precision on ESC DEV25 (run
`26ac5081-c538-431e-994c-b6a7097f7245`). That is a different model/corpus; it
motivates measuring expansion-induced false positives rather than assuming a gain.

## E2 — structured inter-sentence search

Experiment: `77d7ad4d-216b-4d44-b48e-fc7822b586d6`, arm `SEARCH`.

Freeze the historical long vote@5 as the starting prediction. Select at most
24 absent unordered inter-sentence candidate pairs per document, round-robin
over distances 1, 2–3, >=4 (up to eight each when all strata have capacity).
Within a stratum rank by lexical Jaccard overlap in +/-100 character mention
contexts, with hash-based ties. Source text and public sentence/mention metadata
are the only inputs. Selection and its coverage are exploratory, not tuned on gold.
The same candidate list is used by both methods and all five demonstrations.
Blocks contain at most 12 pairs; all pairs require N/A/B judgments.

* `direct`: judge the pairs from the full original text, 6144 thinking +512 JSON.
* `structured`: first predict participant frames and verbatim quotations
  (1024 thinking +2048 JSON), then judge the same pairs using the full text
  plus those frames (3072 thinking +512 JSON).

Both ceilings total 6656 generated tokens per candidate block. Actual tokens,
prompt processing and time are reported separately; equal ceilings do not imply
equal realized compute. A quotation must occur verbatim in the source, contain
the exact event tag and participant string. Reject unsupported frame sets and
continue with empty frames; retain this failure in the statistics. A malformed
final judgment makes the block an empty reading. Invalids are never retried to
obtain a more favorable scientific answer.

Keep historical positives and add a candidate only if >=3/5 passes accept it.
Candidate coverage of missed gold is measured only after freezing/inference;
relations outside the capped pool cannot be recovered by this experiment.
This isolates structured reasoning over a fixed retrieval pool, not an optimized
participant-based retrieval system. It is inspired by KnowQA, not a reproduction:
KnowQA used gold MAVEN-ARG arguments on MAVEN-ERE, whereas these frames are predicted.
[KnowQA, section 4.1](https://www.alphaxiv.org/pdf/2410.04752?page=5).

## Execution

Committed config selects the experiment kind and output arm. The tracked Slurm
CPU controller requests htc_daemon / daemon, 1 CPU, 4G, 24h, license sps.
Its inner srun reserves exactly one L40S, 8 CPU, 96G, 24h, gpu QoS.
All persistent outputs/caches live under `/sps/liris/mcarlesso/maven-recall-20261113`.
Use a private ephemeral API key shared by the server, readiness check and client;
never use or print the container's unrelated key. Isolated ports and cache paths.
Before allocation run regression tests in the pinned SIF, validate the entire
public panel, and hash inputs. Before inference run tokenizer/context, synthetic
45k-token prefill and concurrency checks. Save allocation, utilization, server
and Slurm accounting logs. Score only complete panels.

```sh
python -m unittest steering.test_maven_recall steering.test_maven_budget steering.test_maven_pool
python -m steering.maven_recall prepare
python -m steering.maven_recall run --url http://127.0.0.1:PORT/v1
python -m steering.maven_recall score
```
