# Workere integration, 2026-10-27

The user-supplied kit was imported from `Desktop/eci_alias_next` without modifying
that source directory. `IMPORT_MANIFEST.json` hashes every supplied source file.
The original `reports/` and `examples/out/` remain historical/synthetic records,
not outcomes of this integration or real model predictions.

## Added components

- `eci_alias/workere_prepare.py`: exports the actual 44-document development
  manifest (22 topics, 10,916 unordered candidates), from the pinned local ESC
  Arrow cache. It validates the original demonstration bank order, exact tagged
  source serialization and token spans. Only the 20 protected demonstration
  documents supply model-visible labels. Evaluation gold is in a separate file.
- `eci_alias/workere_run.py`: renders seven frozen historical demonstrations per
  document and A/R condition. The historical YAML and demonstration user/answer
  bytes are preserved; an explicit output adapter overrides final serialization
  and permits registered handles, identically in both arms. Thus this is a NEW
  score/schema comparison, not a byte-identical historical baseline.
- `workere-run.json`: initially authorizes only the first DEV44 document in fixed
  manifest order, 7 A + 7 R, with a scope cap of USD 0.25 and reservations of
  USD 0.01/request. No full pilot is automatically submitted. The USD figure is
  a hard operational ceiling, NOT a measured or estimated invoice.

## Gates and limitations

The original gate incorrectly accepted empty usage and could not detect an
entire omitted document. It now fails closed and supports an expected-document
manifest, provenance hashes, auxiliary bounds and actually served model identity.
Invalid field types cannot bypass or crash pairwise token checks. A/R record and
candidate cardinality are checked. Gold evaluation must cover the whole manifest.

No exact Luna tokenizer was supplied. The adapter instead uses conservative
UTF-8 byte upper bounds, conditional on provider catalog metadata identifying a
GPT byte-level tokenizer, with 1,024 tokens reserved for the four-message framing.
This is NOT exact tokenization. A shared selection prefix is reduced to fit the
768-token conservative auxiliary bound. Actual provider prompt/completion usage
must subsequently pass the <=1% A/R difference and registered ceilings: 16,384
input tokens; completion cap 4,096 in v1 and 16,384 in explicit repair v2.
Completion includes provider-counted reasoning; equality of hidden compute is not
claimed. No truncation, automatic request retry or synchronous model fallback.

Offline rendering currently passes the conservative full-input bound on 17/44
documents. This does not prove that the other 27 exceed the actual token cap.
The first fixed DEV44 document, `1_13ecbplus.xml`, passes, with two selected aliases
from an initial seven, auxiliary byte bound <=756, full-input bound <=13,640.
A full 44-document launch remains blocked by coverage until an exact provider
count/tokenizer or a safe, preregistered budget revision is available.

Provider catalog, credential, token-bound, price and reservation checks precede
paid submission. Requests use the existing tracked Luna batch transport, with
separate cache/ledger and durable receipts. Uncertain POST outcomes must never
be automatically resubmitted. Raw request/response events and usage are retained
on the evaluator side; no credentials enter source or artifacts.

## Local checks and tracked execution

From repository root:

```bash
export PYTHONPATH="$PWD/experiments/eci_alias_next"
uv run --project benchmark --locked --extra hub python -m unittest discover \
  -s experiments/eci_alias_next/tests -v
```

The imported 37 tests were reproduced before edits. After integration there are
62 passing kit/integration tests initially, then 67 after repair-v2 regressions.
Six inherited Luna regression tests also pass;
the old confirmation test now points at an immutable CONF40 campaign fixture,
not the unrelated mutable R3 active campaign. A synthetic 3-document / 4-arm /
7-slot (84 fake outputs) end-to-end CLI path was rerun locally. Its scores are
intentionally fabricated and have no scientific meaning.

Actual experiment execution uses the inherited project run command
`bash experiments/jev/run.sh` through the tracked launcher on the configured SSH
backend. The new branch dispatches this command to the alias runner; it does not
alter any frozen parent node or invoke the historical 41-job campaign.

The runner stores logs under `logs/alias-followup-tracked/`. It emits a final
status and an archive of its evidence in the persisted run log. A gate failure
is a blocker, not evidence that virtual aliases do or do not improve ECI.

## Observed smoke failure and one bounded repair

Run `04704436-7ff7-4456-bc8c-44d306547de3`, source `0fbf1e5`, completed
14 real requests. The batch-reported total cost is USD 0.01547165. Ten outputs
hit the 4,096-token completion cap (including reasoning): A has only one complete
generation, R has three. Eight truncated outputs spent the entire cap on reasoning.
The largest A/R prompt-token difference was 0.29703%, below the 1% criterion.
The gate correctly BLOCKED on truncation. No interpretable F1 comparison exists.

A reporting bug in v1 nevertheless calculated a mechanical all-zero prediction
score after a blocked gate. That archived score is excluded from scientific
interpretation. Repair v2 makes gate PASS mandatory before aggregation, gold
access or scoring; tests exercise this exclusion directly. Malformed provider
triple-list schemas now fail instead of silently turning into empty predictions.

Only one repair is planned: same document, same A/R selections, exact same
messages and demo hashes, same high reasoning; raise the symmetric completion
cap to 16,384. This is a declared budget revision, not a resubmission of the
original 4,096-token requests or a merge of old and new votes. The shared ledger
retains the first run's cost and the cumulative USD 0.25 limit. At the observed
catalog prices, the conservative non-batch upper price is USD 0.0098304 per
request; each request still reserves USD 0.01, with prices rechecked live before
submission. At most 14 additional requests (28 cumulative) and no full DEV44
pilot. If this second smoke is still incomplete, stop and request a decision.
