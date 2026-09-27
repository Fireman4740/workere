# workere

LLM event causality identification (ECI): given a document with tagged event mentions, predict
which mention pairs are causally related (`causes` / `causedby` / `norel`). A rewrite of
agentere with the same capabilities, the same config keys, the same prompts and the same report
schema — so agentere queue files, MLflow params and analysis scripts all carry over.

## Run

```bash
export OPENROUTER_API_KEY=...
python run.py                                   # config.yaml as it is
python run.py --set experiment.resampling.n_runs=5 --set datasets.maven_ere.max_examples=20
python run.py --queue queues/example.yaml       # agentere queue files work unchanged
python run.py --resume logs/runs/<stem>         # finish a crashed run; --retry-failed too
python run.py --from-agentere ../agentere/logs/allatonce/run_<...>.config.yaml
python run.py --dry-run                         # prompts, demo plan, call count; no calls
```

Everything is driven by `config.yaml` (commented key by key). A key it does not declare is an
error, and enumerated values are checked before any document is loaded.

Each run gets `logs/runs/<stem>/`: the report `<stem>.json`, the exact `<stem>.config.yaml`,
`<stem>.traces.jsonl` (one line per LLM attempt, tagged with document, pass and role),
`docs.jsonl` (the checkpoint, one line per finished document), `graphs/` and `figures/`.
MLflow params are logged when the run starts, metrics when it ends.

## Analyse

```bash
python analyze.py sweep 85dccc90                 # vote-threshold sweep, all / intra / inter
python analyze.py sweep 21f411ff --coref         # + oracle coreference propagation
python analyze.py sweep 4c5d68e6 --coref --closure --undirected
python analyze.py rescore RUN                    # vote_to_binary, OR / AND, recompute from rows
python analyze.py ranking RUN [RUN ...]          # votes as a ranker vs TF-IDF / embedding / random
python analyze.py runs | ablations | variance | latex     # the MLflow history (both frameworks)
python analyze.py graph RUN DOC_ID | traces RUN | spend RUN | hist RUN
```

A RUN is a report path, a run directory, a run-id fragment or an MLflow uuid prefix; agentere's
runs (`../agentere/logs/allatonce`, `../agentere/mlflow.db`) are found and read, never written.

## Layout

```
run.py, analyze.py     the two entry points
config.yaml            every setting
prompts/               prompt files (copied verbatim from agentere)
few_shot_rationales/   hand-written demo rationales
queues/                queue files
dataprep/              the standard-eci datasets built from source (notes/DATAPREP_AUDIT.md, notes/ESC_DATAPREP_AUDIT.md):
                       esc_standard_eci.py (direction-corrected; PLOT_LINK / coreference layers),
                       ctb_standard_eci.py (CAT, as the literature reads it; all CLINKs with signals),
                       maven_ere_standard_eci.py (+ CAUSE/PRECONDITION type, coreference, subevents)
                       publish.py: rebuild all three, push to <account>/<name>-workere
eci/                   the package — one module per stage, in run order:
  config   data   syntax   chains   prompts   fewshot   llm   tools
  parse   voting   scope   metrics   report   tracking   runner   graphs
  posthoc/ runs  sweep  rescore  ranking  mining  traces
notes/                 REPORT.md (the refactor), the dataprep audits
tests/                 unit, offline pipeline (scripted model), and parity with agentere's runs
```

## Tests

```bash
python -m unittest tests.test_units tests.test_pipeline -v
python -m unittest tests.test_parity -v    # needs ../agentere/logs; checks five recorded runs
python -m unittest tests.test_esc_dataprep tests.test_ctb_dataprep tests.test_maven_dataprep -v   # needs the sources
```

`test_parity` configures workere from five agentere runs' own config snapshots and requires
byte-identical prompts and demonstrations, the same CoT cache keys, and the logged metrics
reproduced exactly from the recorded responses.

## Dependencies

A subset of agentere's environment: `langchain-core`, `langchain-openai`, `openai`, `datasets`,
`mlflow`, `pyyaml`, `numpy`, `matplotlib`, `graphviz`; optional per feature: `spacy` +
`en-core-web-sm` (syntax), `scikit-learn` (similarity selection, ranking), `sentence-transformers`
+ `torch` (bert selection, ranking), `pandas` (`runs --xlsx`). `langgraph` is no longer needed.
No manifest is shipped: run it with agentere's environment, or declare these in your own.

## Use
```bash
# Install dependencies
uv sync
```
```bash
# Set credentials (required)
read -s OPENROUTER_API_KEY && export OPENROUTER_API_KEY
```
```bash
uv run run.py --queue queues/baseline_coref.yaml --only r3_ctb r3_esc r3_maven
uv run run.py --queue queues/baseline_coref.yaml --only r7_ctb r7_esc
uv run run.py --queue queues/baseline_coref.yaml --only r7_maven 
```
```bash
uv run mlflow server --host 0.0.0.0 --backend-store-uri sqlite:///mlflow.db --port 5000
```