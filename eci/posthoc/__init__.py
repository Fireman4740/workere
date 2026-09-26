"""Re-scoring finished runs, with zero model calls.

  runs     locate and load a run (workere or agentere report) into per-document vote tables
  sweep    vote-threshold sweep x {coreference propagation, transitive closure}
  rescore  vote_to_binary / OR / AND re-scoring, and a full metric recompute from the rows
  ranking  resampling votes as a cause-retrieval ranker against similarity baselines
  mining   the MLflow history: run table, single-param ablations, re-run variance, LaTeX
  traces   reading trace files, per-document token spend

This replaces ~15 agentere scripts that each re-implemented "load run -> tallies -> threshold
-> P/R/F1" with their own subtly different conventions (see sweep.py for the one that matters).
"""
