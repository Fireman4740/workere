#!/usr/bin/env python
"""R2: EventStoryLine TEST-S40 prompts and scoring against the round-3 baselines.

prompts: one row per (doc, pass); each pass shows one gold-answer demonstration drawn from
         an ESC train document of another topic (distinct demos across passes).
score:   plurality vote over Qwen passes (unanswered pass = abstains), binary F1 over the
         17,958 ordered candidate pairs, raw and +coref (oracle ECB+ chains, max over every
         mention pair linking the same two chains, as in round 3), paired doc bootstrap vs
         Luna 7-pass (>= 4/7) and the frozen Luna x d1 fusion (mean of symmetrised scores >= 0.39).
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data" / "esc"


def load() -> tuple[list[dict], dict, dict]:
    docs = json.loads((DATA / "test-s40.json").read_text())
    base = json.loads((DATA / "baseline.json").read_text())
    prompt = json.loads((DATA / "prompt.json").read_text())
    return docs, base, prompt


def build_prompts(passes: int) -> list[dict]:
    docs, _, prompt = load()
    rows = []
    for doc in docs:
        topic = doc["id"].split("_")[0]
        pool = [d for d in prompt["demos"] if d["topic"] != topic]
        chosen = random.Random(f"esc-r2::{doc['id']}").sample(pool, passes)
        for p, demo in enumerate(chosen):
            template = prompt["user_template"]
            rows.append({"doc_id": doc["id"], "pass": p, "demo": demo["id"], "messages": [
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": template.format(doc_text=demo["text"])},
                {"role": "assistant", "content": demo["answer"]},
                {"role": "user", "content": template.format(doc_text=doc["text"])},
            ]})
    return rows


def keys_and_gold(docs, base):
    keys, y = [], []
    for doc in docs:
        gold = set(doc["gold"])
        for pair in sorted(base[doc["id"]]):
            keys.append((doc["id"], *pair.split(",", 1)))
            y.append(pair in gold)
    return keys, np.array(y)


def sym(keys, v):
    idx = {k: i for i, k in enumerate(keys)}
    return (v + v[[idx[(k[0], k[2], k[1])] for k in keys]]) / 2


def propagate(keys, score, docs):
    member = {d["id"]: {m: f"c{i}" for i, ch in enumerate(d["chains"]) for m in ch} for d in docs}
    buckets = defaultdict(list)
    for i, (doc, a, b) in enumerate(keys):
        ga, gb = member[doc].get(a, "s" + a), member[doc].get(b, "s" + b)
        if ga != gb:
            buckets[(doc, frozenset((ga, gb)))].append(i)
    out = score.copy()
    for idx in buckets.values():
        out[idx] = score[idx].max()
    return out


def f1(tp, fp, fn):
    return 100 * 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def summary(pred, y):
    tp, fp, fn = int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum())
    return {"P": round(100 * tp / max(tp + fp, 1), 2), "R": round(100 * tp / max(tp + fn, 1), 2),
            "F1": round(f1(tp, fp, fn), 2), "tp": tp, "fp": fp, "fn": fn}


def paired(a, b, y, docs, draws=2000, seed=42):
    rng, uniq = random.Random(seed), sorted(set(docs))
    def c(p, m): return np.array([(p[m] & y[m]).sum(), (p[m] & ~y[m]).sum(), (~p[m] & y[m]).sum()])
    per = {d: (c(a, docs == d), c(b, docs == d)) for d in uniq}
    out = sorted(f1(*sum(per[d][0] for d in s)) - f1(*sum(per[d][1] for d in s))
                 for s in ([rng.choice(uniq) for _ in uniq] for _ in range(draws)))
    return {"delta": round(f1(*c(a, docs == docs)) - f1(*c(b, docs == docs)), 2),
            "ci95": [round(out[int(.025 * draws)], 2), round(out[int(.975 * draws) - 1], 2)],
            "p_delta_gt_0": round(sum(v > 0 for v in out) / draws, 3)}


def qwen_scores(gen_path: Path, keys) -> tuple[np.ndarray, dict]:
    from eci.parse import ParseError, parse_triples

    passes = defaultdict(list)
    for line in gen_path.read_text().splitlines():
        row = json.loads(line)
        pred = None
        if row.get("finish_reason") != "length" and "</think>" in row["text"]:
            try:
                pred = {(a, b) for a, lab, b in parse_triples(row["text"]) if lab != "norel"}
            except (ParseError, ValueError):
                pred = None
        passes[row["doc_id"]].append(pred)
    frac = np.zeros(len(keys))
    for i, (doc, a, b) in enumerate(keys):
        answered = [p for p in passes[doc] if p is not None]
        frac[i] = sum((a, b) in p for p in answered) / len(answered) if answered else 0.0
    stats = {"passes_per_doc": max(len(v) for v in passes.values()),
             "unanswered": sum(p is None for v in passes.values() for p in v)}
    return frac, stats


def score(gen_path: Path) -> dict:
    docs, base, _ = load()
    keys, y = keys_and_gold(docs, base)
    dids = np.array([k[0] for k in keys])
    luna = np.array([base[k[0]][f"{k[1]},{k[2]}"][0] for k in keys])
    d1 = np.array([base[k[0]][f"{k[1]},{k[2]}"][1] for k in keys])
    fusion_score = (sym(keys, luna) + sym(keys, d1)) / 2
    systems = {"Luna 7-pass >=4/7": luna >= 4 / 7 - 1e-9, "Luna x d1 fusion (frozen)": fusion_score >= 0.39}
    report = {"pairs": len(keys), "positives": int(y.sum())}
    if gen_path is not None:
        frac, stats = qwen_scores(gen_path, keys)
        report["qwen"] = stats
        systems["Qwen vote >=1/2"] = frac >= 0.5 - 1e-9
        systems["Qwen sym vote >=1/2"] = sym(keys, frac) >= 0.5 - 1e-9
        systems["Qwen any pass"] = frac > 0
    coref = {}
    for name, pred in systems.items():
        raw_score = pred.astype(float)
        coref[name + " +coref"] = propagate(keys, raw_score, docs) >= 0.5
    report["raw"] = {k: summary(v, y) for k, v in systems.items()}
    report["coref"] = {k: summary(v, y) for k, v in coref.items()}
    if gen_path is not None:
        for ref in ("Luna 7-pass >=4/7", "Luna x d1 fusion (frozen)"):
            for q in ("Qwen vote >=1/2", "Qwen sym vote >=1/2"):
                report[f"{q} vs {ref}"] = paired(systems[q], systems[ref], y, dids)
                report[f"{q} +coref vs {ref} +coref"] = paired(coref[q + " +coref"], coref[ref + " +coref"], y, dids)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prompts")
    p.add_argument("--passes", type=int, default=3)
    p.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("--gen")
    s.add_argument("--out")
    args = parser.parse_args()
    if args.cmd == "prompts":
        rows = build_prompts(args.passes)
        Path(args.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        print(f"{len(rows)} ESC prompts -> {args.out}")
    else:
        report = score(Path(args.gen) if args.gen else None)
        if args.out:
            Path(args.out).write_text(json.dumps(report, indent=1))
        for block in ("raw", "coref"):
            for name, m in report[block].items():
                print(f"{block:5s} {name:34s} F1={m['F1']:6.2f} P={m['P']:6.2f} R={m['R']:6.2f} TP={m['tp']} FP={m['fp']} FN={m['fn']}")
        for k, v in report.items():
            if " vs " in k:
                print(f"{k:70s} dF1={v['delta']:+.2f} [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}] P(>0)={v['p_delta_gt_0']}")


if __name__ == "__main__":
    main()
