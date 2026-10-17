"""Materialize round-3 TEST-S40 suites and configs from the label-free selection (no metrics)."""
import copy, json, sys
from pathlib import Path

import yaml
from datasets import load_dataset

from benchmark.io import fingerprint
from benchmark.suite import documents, freeze

ROOT = Path("experiments/jev")
R3 = ROOT / "round3"
BUDGET = {"path": "~/.local/share/openresearch/workere-decisions/plan-budget.sqlite",
          "scope": "esc-round3-20261017-10usd", "limit_usd": 10, "initial_spend_usd": 0}


def build(variant="t0", luna_overrides=None, include_d1=True, campaign_name="active.yaml"):
    sel = json.loads((R3 / "round3-selection.json").read_text())
    ids = [d["id"] for d in sel["test_s40"]]
    template = json.loads((ROOT / "data/conf40-12_10ecbplus.xml-suite.json").read_text())
    cfg = copy.deepcopy(template["config"])
    rows = list(load_dataset("Nofing/EventStoryLine-0.9-standard-eci-workere", revision=sel["dataset_revision"], split="train"))
    suite = freeze(cfg, rows=rows)
    byid = {d["id"]: d for d in suite["documents"]}
    suite["documents"] = [byid[i] for i in ids]
    suite["selection"] = {"name": "TEST-S40", "seed": sel["seed"], "document_ids": ids, "revision": sel["dataset_revision"]}
    suite.pop("fingerprint"); suite["fingerprint"] = fingerprint(suite)
    assert len(documents(suite)) == 40 and sum(len(d["pair_list"]) for d in suite["documents"]) == sel["counts"]["test_pairs"]
    (R3 / "data").mkdir(exist_ok=True)
    (R3 / "test-s40-suite.json").write_text(json.dumps(suite, ensure_ascii=False, indent=1) + "\n")
    for doc in suite["documents"]:
        one = {**suite, "documents": [doc], "selection": {**suite["selection"], "document_ids": [doc["id"]],
               "parent_TEST_S40_fingerprint": suite["fingerprint"]}}
        one.pop("fingerprint"); one["fingerprint"] = fingerprint(one)
        (R3 / f"data/test-s40-{doc['id']}-suite.json").write_text(json.dumps(one, ensure_ascii=False, indent=1) + "\n")
    experiments = []
    if include_d1:
        d1 = yaml.safe_load((ROOT / "d1_noul_dev25.yaml").read_text())
        d1["name"] = "esc-r3-d1ultra-noul-test-s40"
        d1["dataset"] = {"name": "esc", "split": "train", "suite": "test-s40-suite.json"}
        params = d1["pipeline"]["relation_predictor"]["params"]
        params["max_workers"] = 256
        params["client"].update(budget_usd=1.5, max_calls=40000, shared_budget=dict(BUDGET))
        d1["output_dir"] = "../../../logs/jev-runs"
        (R3 / "d1-test-s40.yaml").write_text(yaml.safe_dump(d1, sort_keys=False))
        experiments.append({"config": "round3/d1-test-s40.yaml"})
    luna_dir = R3 / f"luna-{variant}"
    luna_dir.mkdir(exist_ok=True)
    base = yaml.safe_load((ROOT / "luna/conf40-1_13ecbplus.xml.yaml").read_text())
    for i in ids:
        c = copy.deepcopy(base)
        c["name"] = f"esc-r3-{variant}-luna-test-s40-{i.replace('.xml', '')}"
        c["dataset"]["suite"] = f"../data/test-s40-{i}-suite.json"
        client = c["pipeline"]["relation_predictor"]["params"]["client"]
        client.update(cache_path=f"~/.local/share/openresearch/workere-decisions/r3-{variant}-luna-{i}.sqlite",
                      shared_budget=dict(BUDGET))
        c["pipeline"]["relation_predictor"]["params"]["path"] = "../../../../prompts/esl_standard_eci_lean_coref.yaml"
        c["output_dir"] = "../../../../logs/jev-runs"
        c["pipeline"]["relation_predictor"]["params"].update(luna_overrides or {})
        (luna_dir / f"test-s40-{i}.yaml").write_text(yaml.safe_dump(c, sort_keys=False))
        experiments.append({"config": f"round3/luna-{variant}/test-s40-{i}.yaml"})
    campaign = {"schema": "workere-campaign/v1", "max_parallel": 9, "experiments": experiments}
    (ROOT / campaign_name).write_text(yaml.safe_dump(campaign, sort_keys=False))
    return campaign


if __name__ == "__main__":
    build()
    print("TEST-S40 materialized: 40 documents; no metrics computed.")
