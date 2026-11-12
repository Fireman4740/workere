# W7/W8: ESC50 and MAVEN50 follow-ups

User requested at least 50 documents per dataset. W7/W8 descend from the frozen
W5/W6 results; neither previous outcome is an independently confirmed winner.

* ESC50: preserve the 40 TEST-S40 documents and add ten other documents from
  the same corpus, selected by sorted identifier outside every demonstration ID.
  Verify original prompts are byte-equivalent, import 53 compatible cached
  readings with original run/commit provenance, and rerun verification. Historical
  P2 scores are available on the original 40 documents, not all 50.
* MAVEN50: retain the original 24 documents and add 26 selected by token-length
  quartile and fixed seed, without using labels. The historical comparator is
  available only on the original 12. Since W6 produced 25/40 invalid readings
  with singleton event IDs, restrict the JSON pair field to the actual MAVEN ID
  format `e<number>,e<number>` for both arms. Do not import W6 readings under
  this changed generation contract. This fixes one observed format failure; it
  does not guarantee correct causal pairs or solve long forced thinking.

Both retain frozen cc0.10, the same five demonstrations per document for both
arms, the 8192-token thinking budget, five+five readings, vote thresholds and
verification policies. Gold is never included in prompts or used to select
documents. Tokenizer/context checks and hook smoke tests run before generation.
ESC50 excludes historical predictions from the common 50-document routing;
historical scores are instead computed on the matched original40 subset. MAVEN
historical scores are likewise restricted to original12. Base5 and cc5 use all
50 documents for their paired comparison. These previously studied samples and
multiple explored variants do not provide independent confirmation.

Sequential on the shared RTX3090: ESC first, then MAVEN. Maximum 36 hours of
new GPU-active time per corpus, checked between batches; an ongoing batch may
overrun. Time waiting for GPU is separate. Previous source observations keep
their original token costs in the reporting. Inputs, cached rows, manifests and
per-batch outputs persist for recovery. Restore the shared Qwen API after both
jobs finish and no other GPU job is using the card.
