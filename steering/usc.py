#!/usr/bin/env python
"""N1/N2 helpers on the 47 controlled CTB docs.

prompts: N2 propose-then-verify. The last user turn of each N13 prompt is extended with the
         causal pairs proposed by earlier independent readings (with how many readings proposed
         each); the model re-checks them against the unchanged annotation rules and answers.
vote:    abstention-aware plurality vote over the conditions/passes of one or more generation
         files (truncated or unparsable pass = abstains), binary F1 on ordered pairs, paired
         doc bootstrap vs the Qwen 7-pass API baseline.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def parsed(row: dict):
    from eci.parse import ParseError, parse_triples

    if row.get("finish_reason") == "length" or "</think>" not in row["text"]:
        return None
    try:
        return [(a, lab, b) for a, lab, b in parse_triples(row["text"]) if lab != "norel"]
    except (ParseError, ValueError):
        return None


def readings(paths: list[str]) -> dict[str, list]:
    out = defaultdict(list)
    for path in paths:
        for line in Path(path).read_text().splitlines():
            row = json.loads(line)
            out[row["doc_id"]].append(parsed(row))
    return out


def build_prompts(src: str, gens: list[str], doc_ids: str) -> list[dict]:
    keep = [d.strip() for d in Path(doc_ids).read_text().splitlines() if d.strip()]
    reads = readings(gens)
    rows = []
    for line in Path(src).read_text().splitlines():
        row = json.loads(line)
        if int(row["pass"]) != 0 or row["doc_id"] not in keep:
            continue
        answered = [r for r in reads.get(row["doc_id"], []) if r is not None]
        votes, labels = defaultdict(int), {}
        for r in answered:
            for pair in {tuple(sorted((a, b))) for a, _, b in r}:
                votes[pair] += 1
            for a, lab, b in r:
                labels.setdefault(tuple(sorted((a, b))), (a, lab, b))
        lines = [f"- {a},{b} ({lab}): proposed by {votes[p]}/{len(answered)} readings"
                 for p, (a, lab, b) in sorted(labels.items(), key=lambda kv: -votes[kv[0]])]
        note = ("\n\nPrevious independent readings of this document proposed the causal pairs below "
                "(with how many readings proposed each). They disagree and can be wrong in both directions.\n"
                + ("\n".join(lines) if lines else "- (no causal pair proposed)")
                + "\nRe-examine every candidate against the annotation rules above: keep a pair only if its "
                  "sentence contains the required lexical anchor; add a pair the readings missed only if the "
                  "text clearly anchors it. Then output the final JSON as specified.")
        messages = [dict(m) for m in row["messages"]]
        messages[-1]["content"] += note
        rows.append({"doc_id": row["doc_id"], "pass": 0, "messages": messages, "candidates": len(lines)})
    return rows


def vote_report(gens: list[str]) -> dict:
    import sys

    sys.path.insert(0, str(HERE))
    from compare import paired_delta

    ref = json.loads((HERE / "data/reference.json").read_text())
    ids = ref["qwen_ids"]
    gold = {d: {tuple(x.split(",", 1)) for x, v in ref["gold"][d].items() if v != "norel"} for d in ids}
    qwen = {d: {tuple(x.split(",", 1)) for x, v in ref["qwen"][d].items() if v != "norel"} for d in ids}
    passes = defaultdict(dict)
    for path in gens:
        for line in Path(path).read_text().splitlines():
            row = json.loads(line)
            r = parsed(row)
            passes[(Path(path).parent.name, row["condition"], int(row["pass"]))][row["doc_id"]] = (
                None if r is None else {(a, b) for a, _, b in r})
    keys = sorted(passes)

    def counts(pred):
        return {d: {"tp": len(pred[d] & gold[d]), "fp": len(pred[d] - gold[d]), "fn": len(gold[d] - pred[d])} for d in ids}

    def f1(c):
        tp, fp, fn = (sum(v[x] for v in c.values()) for x in ("tp", "fp", "fn"))
        return round(100 * 2 * tp / (2 * tp + fp + fn), 2), tp, fp, fn

    base = counts(qwen)
    out = {"passes": ["/".join(map(str, k)) for k in keys]}
    for n in range(1, len(keys) + 1):
        pred = {}
        for d in ids:
            got = [passes[k].get(d) for k in keys[:n] if passes[k].get(d) is not None]
            cnt = defaultdict(int)
            for g in got:
                for p in g:
                    cnt[p] += 1
            pred[d] = {p for p, m in cnt.items() if m / len(got) >= 0.5}
        c = counts(pred)
        out[f"vote@{n}"] = {"f1": f1(c), "vs_qwen7": [round(v, 2) for v in paired_delta(c, base, 10000)]}
    for k in keys:
        out["single " + "/".join(map(str, k))] = {
            "f1": f1(counts({d: passes[k].get(d) or set() for d in ids})),
            "unanswered": sum(passes[k].get(d) is None for d in ids)}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prompts")
    p.add_argument("--src", required=True)
    p.add_argument("--gen", nargs="+", required=True)
    p.add_argument("--doc-ids", required=True)
    p.add_argument("--out", required=True)
    v = sub.add_parser("vote")
    v.add_argument("--gen", nargs="+", required=True)
    v.add_argument("--out")
    args = parser.parse_args()
    if args.cmd == "prompts":
        rows = build_prompts(args.src, args.gen, args.doc_ids)
        Path(args.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        print(f"{len(rows)} propose-then-verify prompts, median candidates "
              f"{sorted(r['candidates'] for r in rows)[len(rows) // 2]} -> {args.out}")
    else:
        report = vote_report(args.gen)
        if args.out:
            Path(args.out).write_text(json.dumps(report, indent=1))
        for name, val in report.items():
            print(f"{name:28s} {val}")


if __name__ == "__main__":
    main()
