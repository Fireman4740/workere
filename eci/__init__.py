"""workere — LLM event causality identification (ECI) pipeline.

The package is organised by stage, in the order a run goes through them:

  config    config.yaml + queue overrides + the active prompt file, validated up front
  data      HuggingFace loading into `Doc` objects
  syntax    optional spaCy block appended after the document text
  chains    event-coreference chains (ESC / ECB+, MAVEN span dataset, surface form)
  prompts   everything that turns a document into chat messages
  fewshot   demo pool, selection, the per-document demo plan, CoT synthesis
  llm       the chat client: retries, timeouts, tool loop, usage ledger, traces
  tools     the agentic tools, bound to one document at a time
  parse     model answer -> relation triples
  voting    resample passes -> one prediction per pair
  scope     mention split, mention whitelist, gold count (prompt addenda + filters)
  metrics   set / multiclass / binary metrics
  report    per-pair rows and the run report (agentere-compatible schema)
  tracking  run directory, checkpoint, git state, MLflow
  runner    the orchestration of one run (and of a queue of runs)
  graphs    causal-graph rendering and per-run histograms
  posthoc   re-scoring finished runs: sweeps, coreference, closure, ranking, mining
"""
