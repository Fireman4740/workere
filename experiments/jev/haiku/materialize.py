"""H1: frozen fusion with Claude Haiku 5.5 (xhigh, batch) x d1-ultra on ESC train233 and Causal-TimeBank 183.

Materializes label-free suites and configs (no metrics, no model call):
  uv run --project benchmark --locked --extra hub python experiments/jev/haiku/materialize.py
"""
import copy
import json
from pathlib import Path

import yaml
from datasets import load_dataset

from benchmark.io import fingerprint
from benchmark.suite import documents, freeze

ROOT = Path("experiments/jev")
H1 = ROOT / "haiku"
BUDGET = {"path": "~/.local/share/openresearch/workere-decisions/plan-budget.sqlite",
          "scope": "haiku-fusion-esc-ctb-20261026", "limit_usd": 12, "initial_spend_usd": 0}
HAIKU = {"model": "anthropic/claude-haiku-5.5:batch", "model_version": "claude-haiku-5.5",
         "served_model_prefixes": ["anthropic/claude-haiku-5.5"], "reasoning_effort": "xhigh",
         "max_output_tokens": 64000, "max_cost_per_call_usd": 0.0175, "budget_usd": 0.15,
         "input_price_per_million": 0.05}
CORPORA = {
    "esc": {"dataset_key": "event_story_line", "dataset_name": "esc",
            "repo_id": "Nofing/EventStoryLine-0.9-standard-eci-workere",
            "revision": "e6418e8747ac1c7fecf66d56ae41fbde7fd5967a", "split": "train",
            "documents": 233, "pairs": 122130, "folds": 5,
            "prompt": "esl_standard_eci_lean_coref.yaml", "demo_exclusion": "topic",
            "historical": {"model": "deepseek/deepseek-v4-flash", "run": "e77431d6", "P": 43.1, "R": 50.8, "F1": 46.6,
                           "status": "reported in notes/eci_sota_to_beat.md, not replayed"}},
    "ctb": {"dataset_key": "causal_timebank", "dataset_name": "causal_timebank",
            "repo_id": "Nofing/CausalTimeBank-standard-eci-workere",
            "revision": "1bba25094d1dda15f172c1f17bb88c26fb3d767d", "split": "train",
            # pair_list of the workere CTB repo used by the historical baseline (19,410 in the non-workere copy)
            "documents": 183, "pairs": 19442, "folds": 10,
            "prompt": "causal_timebank_syntax.yaml", "demo_exclusion": "document",
            "historical": {"model": "deepseek/deepseek-v4-flash", "run": "a43655dd", "P": 65.7, "R": 61.1, "F1": 63.3,
                           "status": "reported in notes/eci_sota_to_beat.md, not replayed"}},
}


def suite_for(key, spec):
    template = json.loads((ROOT / "data/conf40-12_10ecbplus.xml-suite.json").read_text())["config"]
    cfg = copy.deepcopy(template)
    ds = copy.deepcopy(template["datasets"]["event_story_line"])
    ds.update(repo_id=spec["repo_id"], split=spec["split"], max_examples=0,
              kfold={"enabled": True, "n_folds": spec["folds"]})
    if key == "ctb":
        ds["name"] = "CausalTimeBank-standard-eci"
    cfg.update(active_dataset=spec["dataset_key"], datasets={spec["dataset_key"]: ds})
    rows = list(load_dataset(spec["repo_id"], revision=spec["revision"], split=spec["split"]))
    suite = freeze(cfg, rows=rows)
    suite["documents"] = sorted(suite["documents"], key=lambda d: d["id"])
    suite["selection"] = {"name": f"H1-{key.upper()}-ALL", "revision": spec["revision"],
                          "document_ids": [d["id"] for d in suite["documents"]]}
    suite.pop("fingerprint")
    suite["fingerprint"] = fingerprint(suite)
    n_pairs = sum(len(d["pair_list"] if isinstance(d["pair_list"], list) else eval(d["pair_list"]))
                  for d in suite["documents"])
    assert len(documents(suite)) == spec["documents"] and n_pairs == spec["pairs"], (key, n_pairs)
    return suite


def main():
    (H1 / "data").mkdir(parents=True, exist_ok=True)
    (H1 / "configs").mkdir(exist_ok=True)
    experiments, corpora = [], {}
    d1_template = yaml.safe_load((ROOT / "d1_noul_dev25.yaml").read_text())
    doc_template = yaml.safe_load((ROOT / "luna/conf40-1_13ecbplus.xml.yaml").read_text())
    for key, spec in CORPORA.items():
        suite = suite_for(key, spec)
        (H1 / f"data/{key}-suite.json").write_text(json.dumps(suite, ensure_ascii=False) + "\n")
        d1 = copy.deepcopy(d1_template)
        d1["name"] = f"{key}-h1-d1ultra-noul"
        d1["dataset"] = {"name": spec["dataset_name"], "split": spec["split"], "suite": f"../data/{key}-suite.json"}
        params = d1["pipeline"]["relation_predictor"]["params"]
        params["max_workers"] = 256
        params["client"].update(budget_usd=5.0 if key == "esc" else 1.5,
                                max_calls=int(spec["pairs"] * 1.25), shared_budget=dict(BUDGET))
        d1["output_dir"] = "../../../../logs/jev-runs"
        (H1 / f"configs/{key}-d1.yaml").write_text(yaml.safe_dump(d1, sort_keys=False))
        experiments.append({"config": f"haiku/configs/{key}-d1.yaml"})
        for doc in suite["documents"]:
            one = {**suite, "documents": [doc],
                   "selection": {**suite["selection"], "document_ids": [doc["id"]], "parent_fingerprint": suite["fingerprint"]}}
            one.pop("fingerprint")
            one["fingerprint"] = fingerprint(one)
            (H1 / f"data/{key}-{doc['id']}-suite.json").write_text(json.dumps(one, ensure_ascii=False) + "\n")
            c = copy.deepcopy(doc_template)
            c["name"] = f"{key}-h1-haiku-{doc['id'].replace('.xml', '')}"
            c["dataset"] = {"name": spec["dataset_name"], "split": spec["split"], "suite": f"../data/{key}-{doc['id']}-suite.json"}
            c["models"] = [{"id": HAIKU["model"], "version": HAIKU["model_version"]}]
            p = c["pipeline"]["relation_predictor"]["params"]
            p["client"].update(HAIKU, cache_path=f"~/.local/share/openresearch/workere-decisions/h1-haiku-{key}-{doc['id']}.sqlite",
                               shared_budget=dict(BUDGET))
            p.update(path=f"../../../../prompts/{spec['prompt']}", repo_id=spec["repo_id"], revision=spec["revision"],
                     demo_exclusion=spec["demo_exclusion"], passes=7)
            c["output_dir"] = "../../../../logs/jev-runs"
            (H1 / f"configs/{key}-haiku-{doc['id']}.yaml").write_text(yaml.safe_dump(c, sort_keys=False))
            experiments.append({"config": f"haiku/configs/{key}-haiku-{doc['id']}.yaml"})
        corpora[key] = {"d1": f"{key}-h1-d1ultra-noul", "votes": f"{key}-h1-haiku-", "documents": spec["documents"],
                        "historical": spec["historical"]}
    campaign = {"schema": "workere-campaign/v1", "max_parallel": 40, "experiments": experiments,
                "fusion_evaluation": {"corpora": corpora}}
    (ROOT / "active.yaml").write_text("# Fixed launch entry. H1: frozen fusion, Claude Haiku 5.5 xhigh batch x d1-ultra,\n"
                                      "# ESC v0.9 train 233 (all ordered pairs) + Causal-TimeBank 183 (intra pair_list).\n"
                                      + yaml.safe_dump(campaign, sort_keys=False))
    print(f"H1 materialized: {len(experiments)} configs; no metrics computed.")


if __name__ == "__main__":
    main()
