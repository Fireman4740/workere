"""Parity with agentere, measured on agentere's own recorded runs (read-only).

For each run below, workere is configured from the run's config snapshot and must

  1. build byte-identical test prompts (the user message of every inference call in the trace),
  2. build byte-identical demonstration prompts, and find their CoT under the SAME cache key in
     agentere's cot_cache.json, with the same content the trace shows,
  3. reproduce the run's logged metrics by re-parsing the recorded responses and voting them.

Skipped when the agentere logs are not there. AGENTERE=/path/to/agentere overrides the location.
Run: python -m unittest tests.test_parity -v
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

import yaml

from eci import parse, report as R, scope, voting
from eci.config import DEFAULT_CONFIG, load_config, n_passes, overrides_from_agentere
from eci.data import load_eval_docs
from eci.fewshot import FewShotBank
from eci.labels import BINARY_LABELS
from eci.runner import Runner
from eci.tracking import RunDir

AGENTERE = Path(os.environ.get("AGENTERE", Path(__file__).resolve().parents[2] / "agentere"))
LOGS = AGENTERE / "logs" / "allatonce"
MARKERS = ("The correct label assignments for this document are:",
           "Continue. The correct labels are still:",
           "Rewrite your response from scratch to remove all privileged knowledge.")

# ESC whitelist + syntax, MAVEN whitelist + gold count, CTB 10-fold, ESC full baseline.
RUNS = os.environ.get("PARITY_RUNS", "85dccc90 c7ea8631 be86d19c 21f411ff 762e4506").split()

# Runs made by agentere's experiment scripts that patched the pipeline at runtime; in workere
# the same behaviour is a config switch.
EXTRA = {"762e4506": {"coref_hint": {"enabled": True, "source": "esc_ecb",
                                     "esl_root": str(AGENTERE / "data" / "EventStoryLine")}}}


def agentere_run(run_id: str):
    stem = next(iter(sorted(LOGS.glob(f"run_*_{run_id}.json")))).name[:-5]
    snap = yaml.safe_load((LOGS / f"{stem}.config.yaml").read_text())
    logged = json.loads((LOGS / f"{stem}.json").read_text())["results"]
    return stem, snap, logged


def workere_config(snap: dict, extra: dict) -> dict:
    base = yaml.safe_load(DEFAULT_CONFIG.read_text())
    ov, _ = overrides_from_agentere(snap, base)
    ov.setdefault("mlflow", {})["enabled"] = False
    # Read agentere's CoT cache in place; the test never writes it.
    ov.setdefault("few_shot", {}).setdefault("cot_generation", {})["cache_path"] = str(AGENTERE / "logs" / "cot_cache.json")
    ov.setdefault("llm_cache", {})["enabled"] = False
    ov["coref_hint"] = {"enabled": False}
    ov.update(extra)
    return load_config(DEFAULT_CONFIG, ov)


def inference_lines(stem: str):
    """(final user message, demo (human, ai) pairs, answer text) per inference call."""
    out = []
    with open(LOGS / f"{stem}.traces.jsonl", encoding="utf-8") as f:
        for line in f:
            msgs = json.loads(line)
            kinds = [m["id"][-1] for m in msgs]
            if kinds[-1] != "AIMessage" or len(msgs) < 3 or kinds[-2] != "HumanMessage":
                continue
            humans = [m["kwargs"]["content"] for m in msgs if m["id"][-1] == "HumanMessage"]
            if any(mk in h for h in humans for mk in MARKERS):
                continue
            body = msgs[1:-2]
            demos = [(body[i]["kwargs"]["content"], body[i + 1]["kwargs"]["content"])
                     for i in range(0, len(body) - 1, 2)]
            out.append((msgs[-2]["kwargs"]["content"], demos, msgs[-1]["kwargs"]["content"]))
    return out


@unittest.skipUnless(LOGS.is_dir(), f"agentere logs not found at {LOGS}")
class Parity(unittest.TestCase):
    maxDiff = None

    def check_run(self, run_id: str):
        stem, snap, logged = agentere_run(run_id)
        cfg = workere_config(snap, EXTRA.get(run_id, {}))
        runner = Runner(cfg, RunDir(Path(tempfile.mkdtemp()) / "parity"), factory=lambda *a: None)
        docs = load_eval_docs(cfg)
        runner.decorate(docs)
        wl = cfg["mention_whitelist"]

        def test_prompt(doc):
            keep = scope.whitelist_ids(doc, wl["mode"], wl["path"]) if wl["enabled"] else None
            wl_block = scope.whitelist_instruction(keep, doc) if keep is not None else None
            gc = scope.count_instruction(scope.gold_count(doc)) if cfg["gold_count"]["enabled"] else None
            return runner._user_prompt(doc, doc.pair_list, None, wl_block, gc)

        prompt_to_doc = {test_prompt(d): d for d in docs}
        lines = inference_lines(stem)
        by_doc = defaultdict(list)
        unmatched = 0
        for user, demos, answer in lines:
            d = prompt_to_doc.get(user)
            if d is None:
                unmatched += 1
            else:
                by_doc[d.id].append((demos, answer))
        # 1. every document's prompt is found verbatim, and no inference call is unexplained
        self.assertEqual(unmatched, 0, f"{run_id}: {unmatched} inference calls match no workere prompt")
        self.assertEqual(set(by_doc), {d.id for d in docs}, f"{run_id}: documents without a matching call")

        # 2. demos: same human text, CoT found under workere's key with the traced content
        if cfg["few_shot"]["enabled"]:
            bank = FewShotBank(cfg, runner.prompt)
            bank.load(runner.decorate)
            bank.load_cache()
            human_to_demo = {bank.demo_input(p)[0]: p for p in bank.pool}
            key_of_text = {v[0]: k for k, v in bank.cot.items()}
            demo_checked, not_persisted = 0, set()
            for calls in by_doc.values():
                for demos, _ in calls:
                    for human, ai in demos:
                        pool_doc = human_to_demo.get(human)
                        self.assertIsNotNone(pool_doc, f"{run_id}: a traced demo matches no pool document")
                        key = bank.cot_key(pool_doc, *bank.demo_input(pool_doc))
                        if ai in key_of_text:
                            # wherever agentere stored this CoT, it stored it under our key
                            self.assertEqual(key_of_text[ai], key, f"{run_id}: CoT key mismatch")
                        else:
                            # synthesized on the fly during inference: agentere never saved those
                            not_persisted.add(key)
                        demo_checked += 1
            print(f"  {run_id}: {demo_checked} demo turns byte-identical; CoT keys match agentere's "
                  f"({len(not_persisted)} distinct CoTs were never persisted by agentere)")

        # 3. metrics from the recorded answers
        n = n_passes(cfg)
        self.assertFalse(wl["enabled"] and wl["enforce"], "enforced whitelists are not replayed here")
        results = []
        for d in docs:
            passes = []
            for _, answer in by_doc[d.id]:
                try:
                    passes.append(parse.parse_triples(answer, normalize_labels=False))
                except parse.ParseError:
                    continue      # a failed attempt, retried in the original run
            self.assertEqual(len(passes), n, f"{run_id}/{d.id}: {len(passes)} parseable passes")
            if n > 1:
                final, stats = voting.aggregate(passes, cfg["experiment"]["resampling"]["tie_breaking"], None)
            else:
                final, stats = passes[0], voting.single_pass_stats(passes[0])
            results.append({"id": d.id, "doc_idx": d.idx, "lang": d.lang, "gold_triples": d.gold,
                            "pred_triples": final, "pair_stats": stats, "justifications": {},
                            "mention_sentence": d.sentence_of, "first_token": d.first_token,
                            "surfaces": d.surfaces, "failure": None})
        labels = BINARY_LABELS if cfg["data"]["binary_undirected"] else list(cfg["data"]["labels"])
        rep = R.build(results, len(docs), cfg, labels)
        for key in ("micro_f1", "macro_f1", "micro_precision", "micro_recall", "total_pairs",
                    "sentence_unknown_pairs"):
            self.assertAlmostEqual(rep[key], logged[key], places=12, msg=f"{run_id}: {key}")
        for scope_ in ("binary", "binary_intra", "binary_inter"):
            for k in ("precision", "recall", "f1", "support_pos"):
                self.assertAlmostEqual(rep[scope_][k], logged[scope_][k], places=12, msg=f"{run_id}: {scope_}.{k}")
        print(f"  {run_id}: {len(docs)} docs, {len(lines)} inference calls — prompts identical, "
              f"binary F1 {rep['binary']['f1']:.4f} = logged {logged['binary']['f1']:.4f}, "
              f"intra {rep['binary_intra']['f1']:.4f} = {logged['binary_intra']['f1']:.4f}")


def _make(run_id):
    def test(self):
        self.check_run(run_id)
    return test


for _rid in RUNS:
    setattr(Parity, f"test_{_rid}", _make(_rid))


if __name__ == "__main__":
    unittest.main()
