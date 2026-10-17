"""R3-T2: per-document suites + Luna coreference configs for POOL-80 and TEST-S40 (no metrics)."""
import copy, json
from pathlib import Path

import yaml
from datasets import load_dataset

from benchmark.io import fingerprint
from benchmark.suite import freeze

ROOT, R3 = Path("experiments/jev"), Path("experiments/jev/round3")
BUDGET = {"path": "~/.local/share/openresearch/workere-decisions/plan-budget.sqlite",
          "scope": "esc-round3-20261017-10usd", "limit_usd": 10, "initial_spend_usd": 0}
sel = json.loads((R3 / "round3-selection.json").read_text())
plan_sel = json.loads((ROOT / "document-selection.json").read_text())
sets = {"dev25": ("dev", [d["id"] for d in plan_sel["dev_full"]]),
        "d15": ("train", plan_sel["diagnostic_d15"]),
        "conf40": ("train", [d["id"] for d in plan_sel["confirmation"]["docs"]]),
        "test-s40": ("train", [d["id"] for d in sel["test_s40"]])}
template = json.loads((ROOT / "data/conf40-12_10ecbplus.xml-suite.json").read_text())
luna = yaml.safe_load((ROOT / "luna/conf40-1_13ecbplus.xml.yaml").read_text())
(R3 / "coref").mkdir(exist_ok=True); (R3 / "data").mkdir(exist_ok=True)
experiments = []
for name, (split, ids) in sets.items():
    cfg = copy.deepcopy(template["config"])
    cfg["datasets"][cfg["active_dataset"]]["split"] = split
    rows = list(load_dataset("Nofing/EventStoryLine-0.9-standard-eci-workere", revision=sel["dataset_revision"], split=split))
    suite = freeze(cfg, rows=rows)
    byid = {d["id"]: d for d in suite["documents"]}
    for i in ids:
        one = {**suite, "documents": [byid[i]], "selection": {"name": f"R3-T2-{name}", "document_ids": [i],
               "revision": sel["dataset_revision"]}}
        one.pop("fingerprint"); one["fingerprint"] = fingerprint(one)
        path = R3 / f"data/coref-{name}-{i}-suite.json"
        path.write_text(json.dumps(one, ensure_ascii=False, indent=1) + "\n")
        c = copy.deepcopy(luna)
        c["name"] = f"esc-r3-t2-coref-{name}-{i.replace('.xml', '')}"
        c["dataset"] = {"name": "esc", "split": split, "suite": f"../data/{path.name}"}
        c["plugins"] = ["pipelines.jev.coref"]
        client = c["pipeline"]["relation_predictor"]["params"]["client"]
        client.update(cache_path=f"~/.local/share/openresearch/workere-decisions/r3-t2-coref-{i}.sqlite",
                      shared_budget=dict(BUDGET), budget_usd=0.1, max_calls=8)
        c["pipeline"]["relation_predictor"] = {"plugin": "luna_coref", "version": "1", "params": {"client": client}}
        c["output_dir"] = "../../../../logs/jev-runs"
        (R3 / f"coref/{name}-{i}.yaml").write_text(yaml.safe_dump(c, sort_keys=False))
        experiments.append({"config": f"round3/coref/{name}-{i}.yaml"})
(ROOT / "active.yaml").write_text(yaml.safe_dump({"schema": "workere-campaign/v1", "max_parallel": 12,
                                                  "experiments": experiments}, sort_keys=False))
print(f"R3-T2 materialized: {len(experiments)} documents; no metrics computed.")
