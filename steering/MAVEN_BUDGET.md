# MAVEN200 reasoning × answer budget, 2026-11-13

Fixed command: `bash steering/run_pilot.sh`. The committed JSON selects an arm.
All four nodes are siblings of `33718e74-4f38-47df-9a9f-aa68c4afba3a`.

| Arm | Experiment | Thinking maximum | Answer maximum | Trace source |
| --- | --- | ---: | ---: | --- |
| A | edc58a52-310a-456c-8dd0-ff93ea264816 | 20480 | 2048 | fresh |
| B | fbfe4f27-c266-473a-b0d2-a92b85e9b2f3 | 20480 | 4096 | exact A traces |
| C | 79f493a7-a02d-4a3a-b6ef-3777e459c83f | 40960 | 2048 | fresh |
| D | 25c6e99c-60df-4033-a45a-058e22636490 | 40960 | 4096 | exact C traces |

Each arm reserves exactly one L40S (8 CPU, 96 GB RAM, 24 hours) through
`srun` inside a tracked `htc_daemon` CPU controller. A and C can run concurrently;
B follows A, D follows C. The controller propagates failures and cancellation.
Persistent files, snapshots and logs live under `/sps/liris/mcarlesso`.

## Frozen scientific contract

The five SHA256-pinned historical files select exactly the same 200 documents,
five training demonstrations and prompt messages. Evaluation gold is used only
by scoring. No document selection or prompt tuning is performed in this campaign.
These documents have already been examined; results are development comparisons.

All arms use context 65536, greedy two-stage completions, a natural `</think>`
stop and the historical text-to-token continuation protocol. There is no minimum
reasoning length and no forced continuation. The specified budgets are maxima.
The engine is the existing HyperQwen SIF, batch mode, KV fp8, concurrency 32,
GPU utilization .94, and the batch launcher's 2048-token prefill chunks.
The checkpoint is named W4A16, but the historical batch launcher defaults to
`INT8_ACT=int8` and `INT8_LAYERS=mlp`; these defaults are preserved explicitly.
This campaign does not silently switch to a pure W4A16 activation profile.
Actual model/config/tokenizer files and SIF bytes are SHA256-pinned at preparation.
Before data inference, check tokenizer agreement, context capacity, a 45k-token
synthetic prefill, structured outputs and concurrent requests.

Each of 1000 jobs per arm has an atomic trace and answer file. B/D require the
producer's complete 1000-job panel and a committed `source_completion_sha256`.
They reuse the saved final-prompt token IDs and never regenerate reasoning.
An incomplete run can resume missing work, preserving completed traces/answers.
A completed arm refuses another GPU allocation. Files with mismatched contracts
are rejected. Transport errors remain missing; malformed or length-truncated
answers are empty predictions within the fixed five-read vote.

## Analysis

Primary: D−B micro-F1, symmetric relation presence, vote ≥3 of 5.
Secondary: C−A, B−A, D−C and the interaction (D−C)−(B−A).
Paired bootstrap: 5000 joint document draws, seed 20261112; secondary intervals
are unadjusted. Also report thresholds 1/2/4/5, each individual demonstration,
precision/recall, intra/inter-sentence and distance 1/2–3/≥4, newly recovered
historically missed gold, forced closes, answer truncation and token counts.
Missing reads prohibit completion/scoring. TP/FP/FN are unordered counts;
their F1 equals the historical symmetric directed expansion.

`report.json` distinguishes incremental thinking tokens from the full recipe.
B/D reuse does not make reasoning free: include the corresponding producer's
reasoning cost when comparing full recipes. Inference wall time excludes startup;
GPU allocation duration and resources are retained in per-attempt Slurm accounting.
Interrupted attempts also require accounting when totaling compute costs.

## Execution and continuation

Launch A/C with the fixed command via the tracked Slurm backend, CPU flavor,
host `in2p3`, partition `htc_daemon`, QoS `daemon`, 1 CPU, 4G RAM,
24h timeout, `--licenses=sps`. The script requests the GPU allocation itself.

When A/C completes, read its tracked log and verify the reported `MB_COMPLETE`
hash. On the corresponding B/D branch, set `source_completion_sha256` to that
hash, commit, then launch with the same controller resources. Do not edit a
producer once it has a result. Every consumer validates source files before
GPU allocation. All four complete panels are required for:

```sh
/app/venv/bin/python -m steering.maven_budget compare
```

That CPU-only analysis writes `factorial_report.json` in the campaign root.
Run regression checks with the existing container Python before every allocation:

```sh
python -m unittest steering.test_maven_budget steering.test_maven_pool
```
