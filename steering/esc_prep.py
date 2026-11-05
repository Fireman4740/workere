#!/usr/bin/env python
"""Build the self-contained ESC TEST-S40 pack for R2 (run once on a machine with the
round-3 artifacts and HF access; CPU only).

Writes steering/data/esc/:
  test-s40.json   40 docs: id, text, pair_list, gold (ordered positives), oracle ECB+ chains
  baseline.json   per doc and ordered candidate pair: Luna 7-pass vote fraction and d1 p
  prompt.json     esl_standard_eci_lean_coref system/user template + demo pool
                  (ESC train docs outside TEST-S40, gold-answer demonstrations)
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
from pathlib import Path

import yaml


def scores(runs: Path, prefix: str) -> dict:
    out = {}
    for path in runs.rglob("predictions.jsonl"):
        if path.parent.name.startswith(prefix):
            for row in map(json.loads, open(path)):
                for s in row["metadata"].get("decision_scores", []):
                    out[(row["id"], s["source"], s["target"])] = s["positive_probability"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round3", required=True)
    parser.add_argument("--suites", required=True, help="directory of esc-*-suite.json (ESC train233)")
    parser.add_argument("--prompt-yaml", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-demo-chars", type=int, default=3500)
    args = parser.parse_args()
    from datasets import load_dataset

    r3 = Path(args.round3)
    suite = json.loads((r3 / "test-s40-suite.json").read_text())["documents"]
    ids = [d["id"] for d in suite]
    rows = {r["id"]: r for r in load_dataset("Nofing/EventStoryLine-0.9-standard-eci-workere", split="train",
                                             revision="e6418e8747ac1c7fecf66d56ae41fbde7fd5967a")}
    docs = []
    for d in suite:
        g = ast.literal_eval(d["gold"]) if isinstance(d["gold"], str) else d["gold"]
        r = rows[d["id"]]
        chains = [[r["mentions"][i] for i in cl] for cl in r["coref_clusters"]]
        docs.append({"id": d["id"], "text": d["text"], "pair_list": d["pair_list"],
                     "gold": sorted({f"{a},{b}" for a, lab, b in g if lab != "norel"}), "chains": chains})
    luna, d1 = scores(r3 / "runs", "esc-r3-t0-luna-test-s40"), scores(r3 / "runs", "esc-r3-d1ultra-noul-test-s40")
    base = {}
    for doc in ids:
        keys = sorted(k for k in d1 if k[0] == doc)
        base[doc] = {f"{a},{b}": [luna[(doc, a, b)], d1[(doc, a, b)]] for _, a, b in keys}
    assert sum(len(v) for v in base.values()) == 17958, sum(len(v) for v in base.values())
    prompt = yaml.safe_load(Path(args.prompt_yaml).read_text())
    demos = []
    for path in sorted(glob.glob(str(Path(args.suites) / "esc-*-suite.json"))):
        for d in json.loads(Path(path).read_text())["documents"]:
            if d["id"] in ids or len(d["text"]) > args.max_demo_chars:
                continue
            g = ast.literal_eval(d["gold"]) if isinstance(d["gold"], str) else d["gold"]
            answer = [{"pair": f"{a},{b}", "label": lab} for a, lab, b in g if lab != "norel"]
            if answer:
                demos.append({"id": d["id"], "topic": d["id"].split("_")[0], "text": d["text"],
                              "answer": json.dumps(answer)})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "test-s40.json").write_text(json.dumps(docs))
    (out / "baseline.json").write_text(json.dumps(base))
    (out / "prompt.json").write_text(json.dumps({"system": prompt["system"], "user_template": prompt["user_template"],
                                                 "demos": demos}))
    print(f"{len(docs)} docs, {sum(len(v) for v in base.values())} pairs, "
          f"{sum(len(d['gold']) for d in docs)} gold, {len(demos)} demos")


if __name__ == "__main__":
    main()
