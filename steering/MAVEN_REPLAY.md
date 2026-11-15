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
