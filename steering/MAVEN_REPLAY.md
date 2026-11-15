# MAVEN200: requested historical recipe transfers, 2026-11-15

Authorized by user msg_d4d5bb63. All five siblings descend from budget D,
which established the strongest numerical recipe in the preceding campaign.
Fixed entry point: `bash steering/run_pilot.sh`; every experiment uses its own
tracked controller and at most one L40S at a time. No frozen branch is edited.

These are transfers of the named **selected policies**, not reproductions of
all the original CTB sweeps. Their CTB scores are not MAVEN200 measurements.
Every arm uses the same exposed 200 documents, same five permitted training
demonstrations, frozen checkpoint/SIF, total context65536, KVfp8 and historical
MLP INT8 activation profile. Evaluation gold is excluded from planners/jobs.
No corpus-specific CTB lexical-only rule is added to MAVEN's implicit causation.

| Arm | Experiment | Frozen transfer |
| --- | --- | --- |
| N1 | 801cf0fb-536c-42ea-9732-29faacba223f | Five different demos, greedy,8192thinking+4096JSON; fixed vote>=3/5 |
| N2 | 611e404a-3c80-4a88-bfdb-c444829fa4ab | Five R1 sampled readings of one demo,T=.6,8192+4096, then one greedy8192+4096 document verification with proposal counts |
| VC1 | 08c1351e-6ac9-4471-aca9-1d608aeb7527 | Remove a D-positive only if both valid C2 local screens reject it; all81967candidatepairs, blocks12, original deterministic shuffled presentation order,4096+384 |
| W4 | 3298ff8c-06a9-4c06-81a3-b59d3712b6d9 | Selected historical__base_filter policy: full-text pair blocks8, two event orders,4096+512; remove only paired N; no weight modification |
| MV2 | ad29fc44-b32a-4c62-a4d6-8967cb809245 | Reuse completed VC1 screens; direct2560+512 local judgments both event orders on any valid screen/D disagreement; strict_add_majority_remove |

D's completion SHA256 is frozen in all configs:
`9098b43d14473cc4f6f44cedd43351f590aeb6e21e87aaa1abb88ef0f6b89366`.
Its predictions are reconstructed from the hashed1000 answers, not an unchecked
prediction file. The same historical documents/structure are verified. D's full
C+D production cost remains necessary; consumer calls are incremental.

W4's selected filter did not need edited weights. Its source sweep's routing
depended on two new five-read families. This transfer deliberately replaces that
route with up to24 D-positive pairs/document ranked by D's support closest to
50%, hash ties; this adaptation is frozen before target generation. Unrouted,
invalid or disagreeing checks retain D. We are not rerunning the96-variant sweep
or claiming a new causal weight-edit experiment.

MV2 shares exactly VC1's C2 requests and recorded responses, avoiding duplicate
screening inference. Its CPU controller waits for VC1 to complete before asking
for a GPU. The producer's experiment, D identity, jobs, response hashes and
completion hash are verified/frozen before MV2 inference. With all four valid
checks, an original negative is added only at4/4 positive; an original positive
is retained only when>1/4 positive. Incomplete panels retain D. The same-model
views are correlated. Five-view majority, four-view consensus and the screen
majority are secondary policies reported without choosing a winner on gold.

Each arm's main comparison is its full recipe minus D, micro-F1 of unordered
causal presence on all200 documents. This is **not an equal-budget ablation**.
Paired document bootstrap5000draws,seed20261112, nominal unadjusted95% intervals
for five exploratory comparisons. Also report precision/recall, intra/inter and
distance1/2-3/>=4, invalids, truncation, forced thinking closes and emitted tokens.
N2 also compares to its own R1 vote. When N1 and N2 both complete, compare their
frozen five-reading panels without another GPU call. No held-out claim.

Invalid document answers are empty predictions in a fixed five-read vote.
Invalid pair answers abstain, retaining D. Transport failures remain missing;
partial panels prohibit final scoring. Every thinking/final stage is atomic and
validated by manifest/job contracts and content hashes; resumes never replace
completed observations. No response is regenerated to obtain a better score.

All persistent data live under `/sps/liris/mcarlesso/maven-compare-20261115`.
Controller: htc_daemon,daemon,1CPU,4G,license sps. Inner allocation: oneL40S,
8CPU,96G,24h. Planned leases stop scheduling after18h, drain pending requests,
save every response and return75; the same tracked controller then releases the
GPU and acquires the next single-GPU allocation. Up to6leases per experiment;
otherwise report incomplete, not a partial score. Non-checkpoint errors stop.
Controllers have5day ceilings; MV2 has10days including waiting for VC1.
Every allocation retains separate server/utilization/Slurm accounting logs.

Run existing plus transfer regression checks before every allocation. Perform
model/tokenizer/context/schema and concurrent smoke checks before inference.
The scripts never leave inference processes running on a login node.

Source implementations are pinned by commit in each config: N1 d5060c5,
N2 80649a7, VC1 f6fca6c, W4 8c0d826 and MV2 c28119b. Scientific adaptations
are the new MAVEN source reference, input/prompt contract and routing described
above; historical output numbers are not imported as new results.


## User-authorized RTX3090 migration (2026-11-15)

L40S jobs were blocked with Reserved for maintenance before any GPU inference.
The user explicitly authorized the RTX3090 and HyperQwen batch container.
The same provisional nodes move to SSH, preserving the fixed entry point.
They run sequentially N1 -> N2 -> W4 -> VC1 -> MV2, behind a common GPU lock.
The new container image is pinned to f706b092aabb156c23fe98c312ba7912fb8dbc0c107d4f152382113e33762f1c.
Context remains65536; concurrency16 and GPU utilization0.90 fit24GB.
INT8 MLP activations, FP8 KV and FP16 recurrent states are retained.
Every model/config/tokenizer byte must match the frozen D checkpoint.
The image/vLLM/runtime hashes are recorded separately from D's L40S engine;
new-runtime results are recipe comparisons with a fixed cached D reference,
not a hardware-controlled causal ablation. All five transfers use the same
3090 runtime; the N1/N2 control comparison is on that shared runtime.

Original scientific files remain on SPS. Their required immutable input subset
is streamed directly into a dedicated Windows-mounted volume under
F:/IA/Qwen3.8/server-extensions/steering/maven3090/sps, mounted at the same
/sps/liris/mcarlesso paths inside batch. No model weights are transferred.
New outputs use maven-compare-20261115-3090, preserving the earlier prepared
L40S manifests. Final outputs must be archived back to SPS after completion.
The batch container was recreated after the older idle container lost GPU
access; host and fresh-container CUDA preflights passed. No active API or
scientific process was stopped. No model/image update was downloaded.
# IN2P3 return: 18-hour allocations (2026-11-15 evening)

User request msg_3b1b0dcb authorizes 18-hour L40S allocations. Pending N2,
W4, VC1 and MV2 move to a fresh SPS namespace, with the original L40S engine
identity and concurrency32. N1 keeps its already-running 3090 panel; none of
its partial responses are mixed with L40S generations or regenerated.
N1-versus-N2 is consequently a descriptive recipe comparison with different
hardware/runtime settings, not a controlled test of demonstration diversity.

Each GPU allocation requests 64800 seconds; the worker stops scheduling after
57600 seconds (16h), retains immutable traces and answers, drains work, and
resumes missing requests on the next allocation. HTTP calls time out after
1200 seconds so draining cannot wait indefinitely. Six GPU allocations are
allowed. The tracked CPU controller may wait for allocation availability;
MV2 waits for the validated VC1 completion before requesting a GPU. The
scientific prompts, budgets, routing and scoring are unchanged.

Diagnostics obtained L40S allocations for6h and12h, while18h requests remained
pending with `ReqNodeNotAvail, Reserved for maintenance`. An18h submission is
therefore a queued request, not evidence that scientific inference has begun.

## Superseding request: 12-hour automatic continuation

User msg_b8f873fc requests12h allocations and automatic continuation. N2,W4,
VC1,MV2 now use a fresh `maven-compare-20261115-l40s12h` namespace,12h GPU
allocations and10h of new work per allocation. Immutable completed responses
and intermediate thinking traces survive each rollover; only missing work is
resubmitted. The CPU controller requests another GPU on worker exit75 or a
verified parent-allocation SlurmTIMEOUT, independent of srun's exit code.
It does not automatically retry cancellation,OOM or unrelated failures.
Accounting polling tolerates delayed final Slurm state. There is a ceiling of
12allocations (up to120h of scheduled work) and no score on an incomplete panel.
N1's nearly complete3090 run stays untouched. Scientific recipes and engine
identity are unchanged. The fresh12h diagnostic at20:16CET was pending for
maintenance, unlike the successful12h diagnostic at19:21CET; monitor actual
allocation status rather than treating submission as GPU execution.
