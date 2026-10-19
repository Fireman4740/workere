"""Harness² `BenchmarkAdapter` for document-level event causality identification (ESC).

One task = one article with its reference event mentions. A rollout runs the frozen engine
on that one document with a harness-driven predictor. Engine outputs, including scores
against annotations, stay in a hidden store outside the rollout directory; the rollout
directory holds only the trajectory and the predicted causal pairs.
"""
import copy
import json
import shutil
from pathlib import Path

import yaml
from harness2.core import tree as harness_tree
from harness2.core.spec import BenchPrompts, BenchSpec, ChannelSpec
from harness2.substrate.opencode import OpencodeSubstrate

from benchmark.configuration import load_experiment, seed_configs
from benchmark.io import read_json, read_jsonl
from benchmark.labels import is_norel
from benchmark.protocols import public_inputs
from pipelines.harness2_eci import surface
from pipelines.jev.llm import marked_text

DELIVERABLE, TRAJECTORY = "causal_pairs.md", "trajectory.md"

PROMPTS = {
    "luna": BenchPrompts(
        work=("Read one news article in which every event mention is tagged <ID text>, and list every "
              "ordered pair of mentions where one event causes, enables or prevents the other, with its "
              "direction. A single model request reads the article and answers; it runs seven times with "
              "a different worked example from another news story each time, and a pair is kept when "
              "most of the seven answers contain it."),
        domain_word="news topic", deliverable_word="list of causal pairs",
        inv_levers=(
            "- the always-on text is the whole instruction: systemprompt.md is the system message, "
            "tool_descriptions/tool_guidance.md holds the rules and output format placed right before "
            "the article, guardrails.md / LongTermMEMORY.md / skills are appended to the system message\n"
            "- there are no tools, no subagents and no plugins: the model answers in one shot\n"
            "- scripts/postprocess.py, if present, is run on the seven answers: stdin JSON with "
            "`mentions` (id, surface, sentence) and `pairs` (source, target, `votes` = the seven labels, "
            "`probability` = fraction of causal votes, `default_label`); stdout JSON {\"labels\": {\"a,b\": "
            "label}} for pairs to relabel (causes, causedby, norel). Use it for deterministic "
            "consistency (direction symmetry, vote cut-offs, same-event propagation)\n"
            "- the output format must stay a JSON array of {\"pair\": \"a,b\", \"label\": ...}"),
        dom_levers=(
            "- what counts as cause, enablement or prevention versus temporal sequence, correlation, "
            "or the same event restated\n"
            "- grouping mentions that refer to one event, and propagating a causal link to every "
            "mention of both events\n"
            "- cross-sentence links, reported speech, plans and hypotheticals, negated events"),
        domain_guidance=("Events in one article cluster around a real-world incident (an arrest, an "
                         "earthquake, a shooting, a merger). The articles of one topic report the same "
                         "kind of incident."),
        yardstick=("Every pair the article supports as direct cause or prevention is listed with the "
                   "right direction for every mention of both events, and nothing merely sequential or "
                   "correlated is listed.")),
    "decision": BenchPrompts(
        work=("Read one news article and decide, separately for every ordered pair of tagged event "
              "mentions, whether one event causes the other. Each pair is one independent request to "
              "a decision model that returns only the probability that the answer is true; a pair is "
              "kept when that probability reaches 0.5."),
        domain_word="news topic", deliverable_word="list of causal pairs",
        inv_levers=(
            "- the decision request has exactly three editable parts: systemprompt.md is the question's "
            "instruction; tool_descriptions/tool_guidance.md must be two lines `true: ...` and "
            "`false: ...` describing when to answer each; guardrails.md, LongTermMEMORY.md and skill "
            "bodies are appended to the instruction (6000 characters in total)\n"
            "- the decision model cannot reason step by step or see other pairs: every instruction "
            "must be decidable from the article and the two mentions alone\n"
            "- scripts/postprocess.py, if present, sees every pair's probability at once: stdin JSON "
            "with `mentions` (id, surface, sentence) and `pairs` (source, target, `probability`, `default_label`); stdout JSON {\"labels\": {\"a,b\": \"causal\"|\"norel\"}} "
            "for pairs to relabel. Use it for cut-offs, symmetry across the two orders of a pair, or "
            "propagation across mentions of one event\n"
            "- no tools, subagents or plugins run"),
        dom_levers=(
            "- what counts as cause, enablement or prevention versus temporal sequence, correlation, "
            "or the same event restated\n"
            "- mentions of the same event, subevents, reported speech, plans and negated events"),
        domain_guidance=("Events in one article cluster around a real-world incident (an arrest, an "
                         "earthquake, a shooting, a merger). The articles of one topic report the same "
                         "kind of incident."),
        yardstick=("The kept pairs are exactly those the article supports as a direct causal or "
                   "preventive link, for every mention of both events, and none that are merely "
                   "sequential or correlated.")),
}


class EciAdapter:
    def __init__(self, *, executor, template_path, suites_dir, documents, root, model):
        if executor not in PROMPTS:
            raise ValueError(f"executor must be one of {sorted(PROMPTS)}")
        self.executor, self.documents = executor, list(documents)
        self.template_path, self.suites_dir = Path(template_path).resolve(), Path(suites_dir).resolve()
        self.root = Path(root).resolve()
        self.hidden = self.root / "hidden"
        self.substrate = OpencodeSubstrate()
        channel = ChannelSpec("opencode", model, effort="high", steps=60, timeout=1800)
        self.spec = BenchSpec(name=f"eci-{executor}", runs_root=self.root / "runs",
                              install_mode="workspace_config", prompts=PROMPTS[executor],
                              channels={"proposer": channel, "selector": channel},
                              domain_word="news topic", leak_globs=("*metrics*", "*gold*", "*annot*"))
        self._inputs = {doc: self._load(doc) for doc in self.documents}
        self._write_base()
        for doc in self.documents:
            self._write_source(doc)

    def _template(self):
        return yaml.safe_load(self.template_path.read_text())

    def _suite(self, doc):
        return self.suites_dir / f"{doc}.json"

    def _load(self, doc):
        items = list(public_inputs(read_json(self._suite(doc)), "relations_gold"))
        if len(items) != 1 or items[0].document.id != doc:
            raise ValueError(f"suite for {doc} must hold exactly that one document")
        return items[0]

    def _write_base(self):
        if self.executor == "luna":
            params = self._template()["pipeline"]["relation_predictor"]["params"]
            prompt_path = (self.template_path.parent / params["path"]).resolve()
            base = surface.luna_base_tree(yaml.safe_load(prompt_path.read_text()))
        else:
            base = surface.decision_base_tree()
        surface.validate(base, self.executor)
        harness_tree.write_tree(self.substrate.surface(), base, self.base_harness())

    def _write_source(self, doc):
        item = self._inputs[doc]
        d = self.task_files(doc)
        d.mkdir(parents=True, exist_ok=True)
        rows = [f"| {e.id} | {' '.join(item.document.tokens[i] for i in e.token_indices)} | "
                f"{item.document.sentence_of(e.token_indices)} |" for e in item.reference_entities]
        (d / "article.md").write_text(
            f"# Article {doc}\n\n{marked_text(item.document, item.reference_entities)}\n\n"
            "## Event mentions\n\n| id | text | sentence |\n|---|---|---|\n" + "\n".join(rows) + "\n",
            encoding="utf-8")

    def tasks(self, split="all"):
        return list(self.documents)

    def domain_of(self, task_id):
        return "topic-" + task_id.split("_")[0]

    def task_prompt(self, task_id):
        n = len(self._inputs[task_id].reference_candidates)
        return (f"Article {task_id} (source/article.md) has its event mentions tagged <ID text>. "
                f"Identify which of the {n} ordered mention pairs are causal (one event causes, enables "
                f"or prevents the other) and deliver them as a list of causal pairs.")

    def task_files(self, task_id):
        return self.root / "sources" / task_id

    def base_harness(self):
        return self.root / "base_harness"

    def validate_tree(self, tree):
        surface.validate(tree, self.executor)

    def _config(self, task_id, harness_dir, run_id):
        cfg = copy.deepcopy(self._template())
        cfg["name"] = "h2-" + run_id.replace("/", "__")
        cfg["dataset"]["suite"] = str(self._suite(task_id))
        params = cfg["pipeline"]["relation_predictor"]["params"]
        if "path" in params:
            params["path"] = str((self.template_path.parent / params["path"]).resolve())
        if "selection_path" in params:
            params["selection_path"] = str(Path(params["selection_path"]).resolve())
        params["harness_dir"] = str(harness_dir)
        cfg["output_dir"] = str(self.hidden / run_id / "engine")
        return cfg

    def execute(self, task_id, harness_dir, run_id, *, timeout=None):
        from benchmark.engine import execute as run_engine
        run_dir = self.root / "runs" / "rollouts" / run_id
        if (run_dir / DELIVERABLE).is_file():
            return run_dir
        hidden = self.hidden / run_id
        if hidden.exists():
            shutil.rmtree(hidden)
        snapshot = hidden / "harness"
        shutil.copytree(Path(harness_dir or self.base_harness()), snapshot)
        config_path = hidden / "config.yaml"
        config_path.write_text(yaml.safe_dump(self._config(task_id, snapshot, run_id), sort_keys=False))
        (config,) = list(seed_configs(load_experiment(config_path)))
        result = run_engine(config)
        if result["status"] != "completed":
            raise RuntimeError(f"engine run {result['run_dir']} ended {result['status']}")
        (row,) = read_jsonl(Path(result["run_dir"]) / "predictions.jsonl")
        (hidden / "receipt.json").write_text(json.dumps({"task": task_id, "engine_run_dir": result["run_dir"]}))
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / TRAJECTORY).write_text(row.get("metadata", {}).get("h2_trace", ""), encoding="utf-8")
        (run_dir / DELIVERABLE).write_text(self._render(task_id, row), encoding="utf-8")
        return run_dir

    def _render(self, task_id, row):
        item = self._inputs[task_id]
        doc, ents = item.document, {e.id: e for e in item.reference_entities}

        def mention(mid):
            e = ents[mid]
            return f"{mid} \"{' '.join(doc.tokens[i] for i in e.token_indices)}\" (s{doc.sentence_of(e.token_indices)})"
        kept = [r for r in row["relations"] if not is_norel(r["label"])]
        lines = [f"- {mention(r['source'])} -> {mention(r['target'])}: {r['label']}" for r in kept]
        return (f"# Causal pairs for {task_id}\n\n{len(kept)} of {len(row['relations'])} ordered pairs "
                "kept.\n\n" + ("\n".join(lines) or "(none)") + "\n")

    def delivered(self, run_dir):
        return (Path(run_dir) / DELIVERABLE).is_file()

    def trajectory(self, run_dir, *, obs_trunc):
        text = (Path(run_dir) / TRAJECTORY).read_text(encoding="utf-8")
        return text if len(text) <= obs_trunc else text[:obs_trunc] + "\n[... truncated]"

    def deliverable_text(self, run_dir):
        path = Path(run_dir) / DELIVERABLE
        return path.read_text(encoding="utf-8") if path.is_file() else ""

    def deliverable_files(self, run_dir):
        return {DELIVERABLE: self.deliverable_text(run_dir)}

    def score(self, run_dir, task_id):
        """Binary counts against annotations; read only after selection, from the hidden store."""
        receipt = self.hidden / Path(run_dir).relative_to(self.root / "runs" / "rollouts") / "receipt.json"
        if not receipt.is_file():
            return None
        engine_dir = Path(json.loads(receipt.read_text())["engine_run_dir"])
        (stats,) = read_json(engine_dir / "metrics.json")["document_statistics"]
        return {key: stats["metrics"][key] for key in ("binary", "binary_intra", "binary_inter")}
