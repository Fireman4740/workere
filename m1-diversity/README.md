# M1 diversity experiment

This is the implementation of the authorized C15 versus H7O+7MP pilot, with
unchanged historical fusion and Z0-only union admission. No training is used.

`frozen/m1-protocol-frozen.json` identifies the deterministic 20-document,
20-topic consumed-development sample and all 440 exact requests. The two M0
documents are excluded. The protocol and requests SHA-256 are immutable.

The shared 140 requests run once in a first batch; a second batch executes the
fixed remaining 160 ordinary and140 MP requests. Outputs never condition later
prompts. All channels use high reasoning and a 10000-token inclusive completion
cap. This differs from M0's32768 cap, and bounds marginal API exposure at
2.7269085USD before undocumented batch discounts, below the3USD ceiling.
No retry of a paid POST or invalid/empty/truncated prediction is allowed.

Run the committed experiment via the project's tracked launch interface with
the inherited `bash experiments/jev/run.sh` command. The run controller refuses
direct inference without a run ID. Persistent phase receipts prevent duplicate
payment on retrieval/resume. Use the recorded batch IDs after a monitoring
failure; never submit a new batch to replace an ambiguous prior POST.

Local checks, without model calls:

```sh
uv run --isolated --no-project python m1-diversity/preflight.py --dir m1-diversity/frozen
uv run --isolated --no-project python -m unittest discover -s m1-diversity -p 'test_*.py' -v
```

The existing `eci_m0_pilot` code, data, prompt, parser and historical artifacts
are unchanged. The M1 code imports those pure transformations. The original M0
output semantic hash and reconstructed events/ledgers are checked in
`frozen/m1-m0-audit.json`.

`evaluation-only/controls.json.gz` is a verbatim compressed copy of the
historical control blob, so the immutable remote source needs no Git history.
Only the evaluation script decompresses it, after checking the hash of final
predictions and the protocol. Preparation projects away the gold field when
extracting the historical reference. Selection uses only topic, candidate
counts and technical availability. An accidental partial display of the source
gold file during initial inspection is disclosed in the protocol and report;
this is explicitly exploratory, not a blind confirmation.

The analysis uses a paired topic bootstrap with20000 draws, LOTO, exact
micro-F1 and disjoint channel memberships. Reasoning tokens are already part
of completion tokens and are never added twice. Weighted token cost is an
observable index, not FLOPs or an invented per-system provider invoice.

The separate M1 draft node34b19e46 remained idle, never run and clean in its
owner's worktree at the last prelaunch audit. It was left untouched. This
execution uses node e06a915f-f97c-4f97-8310-bc3338d9b0e4.

After completion, reproduce evaluation without new inference:

```sh
uv run --isolated --no-project python m1-diversity/m1-analysis.py --dir <run-artifacts> --repo <worktree>
```

Do not regenerate documents, masks, prompts, thresholds or allocations after
evaluation. There is no automatic M2.
