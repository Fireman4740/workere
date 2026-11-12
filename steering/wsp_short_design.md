# Reduced W5/W6 pilots, 2026-11-12

User request: relaunch a shorter pilot on fewer documents. The interrupted full
ESC40/MAVEN24 runs were cancelled; these provisional nodes now test four documents
each. CTB47 is complete and remains frozen. No further full-corpus run is queued.

Only corpus size and new active-time ceiling change. Keep the exact existing
five demonstration prompts per document, paired unsteered/cc0.10 reads, 8192-token
thinking budget, mixed vote, and seven verification policies. Verification still
uses both event orders, both judges, 4096 thinking tokens, and at most 24 pairs per
document. The identical generation/scoring modules are hash-checked.

Select ESC's lexicographically first document in each of the first four distinct
topics. Select MAVEN's first two IDs in each of the frozen original12 and extension12
cohorts. Selection uses no labels, scores, or model outputs. These convenience
samples support exploratory debugging only, not a full-benchmark record claim.
Keep every selected document, including documents with no gold or predictions.

The reduced bundles use a new directory and manifest. Frozen source requests are
copied byte-equivalently as JSON objects. Cache import validates the audited raw
cache hash, original manifest, request keys, engine and code identities. Store raw
imported source rows separately; preserve answers, thought, validity, token counts,
run and commit in working copies, changing only manifest metadata and recording
the origin. Imported work remains included in token costs. No verification cache
is imported. Historical baseline predictions and denominators are restricted to
the same documents; MAVEN's historical comparison is confined to original2.

Run ESC then MAVEN under the existing exclusive GPU lock. Each has a cumulative
two-hour ceiling for new active execution, checked between batches (one batch
may overrun); prior cancelled long-run time remains in its separate ledger. Cache
partial work on timeout, never score incomplete observations as a complete pilot.
Both pilots require 80 document readings total instead of 640; 15 ESC readings
are reusable if the audited cache is unchanged. Maximum verification responses:
48 per dataset. The usual topic/document bootstrap is descriptive with four units;
all correlated variants are exploratory, with no automatic promotion.
