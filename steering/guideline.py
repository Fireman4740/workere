#!/usr/bin/env python
"""R3: align the CTB system prompt with the annotation convention, then vote and score.

The N13 prompt sends purpose/intention to norel and says the causative predicate is usually
not an endpoint. On the 133 training documents, purpose/enablement links are 40 of 193 gold
pairs and are found 28% of the time (explicit connectives: 41%). This rewrite keeps every other
rule and the demonstrations unchanged; its examples are generic sentences, not corpus text.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

OLD_PURPOSE = ("Everything else is norel, however sensible the causal story: purpose or intention (in talks to\n"
               "settle the dispute, an attempt to get out), enablement chains you reconstruct yourself")
NEW_PURPOSE = ("Purpose and enablement stated in the sentence ARE causal: an action taken to achieve another\n"
               "event (\"<e1 raised> prices to <e2 offset> costs\": e1 causes e2), and an event that allows,\n"
               "helps, blocks or prevents another (\"a <e3 ruling> <e4 blocking> the <e5 merger>\": e4 causes e5).\n"
               "Everything else is norel, however sensible the causal story: enablement chains you reconstruct yourself")
OLD_BRIDGE = ("That bridging predicate is usually NOT\n"
              "one of the two events you label: label the endpoints filling its cause and effect slots, which\n"
              "often sit two or three mentions apart with other mentions in between — not the adjacent steps.")
NEW_BRIDGE = ("Label the endpoints filling its cause and effect slots, which often sit two or three mentions\n"
              "apart; when the causative predicate is itself a tagged event, it is an endpoint too\n"
              "(\"the offer was <e6 triggered> by the <e7 sale>\": e7 causes e6).")


def rewrite(system: str) -> str:
    for old, new in ((OLD_PURPOSE, NEW_PURPOSE), (OLD_BRIDGE, NEW_BRIDGE)):
        if old not in system:
            raise SystemExit(f"prompt section not found: {old[:60]!r}")
        system = system.replace(old, new)
    return system


def vote_report(gen: Path, k: int) -> dict:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import read_json
    from compare import paired_delta
    from eci.parse import ParseError, parse_triples

    ref = read_json(Path(__file__).resolve().parent / "data/reference.json")
    ids = ref["qwen_ids"]
    gold = {d: {tuple(x.split(",", 1)) for x, v in ref["gold"][d].items() if v != "norel"} for d in ids}
    qwen = {d: {tuple(x.split(",", 1)) for x, v in ref["qwen"][d].items() if v != "norel"} for d in ids}
    answers = defaultdict(dict)
    for line in gen.read_text().splitlines():
        row = json.loads(line)
        pred = None
        if row.get("finish_reason") != "length" and "</think>" in row["text"]:
            try:
                pred = {(a, b) for a, lab, b in parse_triples(row["text"]) if lab != "norel"}
            except (ParseError, ValueError):
                pred = None
        answers[row["condition"]][row["doc_id"]] = pred
    conds = sorted(answers)[:k]

    def counts(pred):
        return {d: {"tp": len(pred[d] & gold[d]), "fp": len(pred[d] - gold[d]), "fn": len(gold[d] - pred[d])} for d in ids}

    def f1(c):
        tp, fp, fn = (sum(v[x] for v in c.values()) for x in ("tp", "fp", "fn"))
        return round(100 * 2 * tp / (2 * tp + fp + fn), 2), tp, fp, fn

    out = {}
    for n in range(1, len(conds) + 1):
        pred = {}
        for d in ids:
            got = [answers[c].get(d) for c in conds[:n] if answers[c].get(d) is not None]
            cnt = defaultdict(int)
            for g in got:
                for p in g:
                    cnt[p] += 1
            pred[d] = {p for p, m in cnt.items() if m / len(got) >= 0.5}
        c = counts(pred)
        out[f"vote@{n}"] = {"f1": f1(c), "vs_qwen7": [round(v, 2) for v in paired_delta(c, counts(qwen), 10000)]}
    for c in conds:
        out[f"single {c}"] = {"f1": f1(counts({d: answers[c].get(d) or set() for d in ids})),
                              "unanswered": sum(answers[c].get(d) is None for d in ids)}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prompts")
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    s = sub.add_parser("vote")
    s.add_argument("--gen", required=True)
    s.add_argument("--k", type=int, default=5)
    s.add_argument("--out")
    args = parser.parse_args()
    if args.cmd == "prompts":
        rows = []
        for line in Path(args.src).read_text().splitlines():
            row = json.loads(line)
            row["messages"] = [{**m, "content": rewrite(m["content"])} if m["role"] == "system" else m
                               for m in row["messages"]]
            rows.append(row)
        Path(args.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        print(f"{len(rows)} rewritten prompts -> {args.out}")
    else:
        report = vote_report(Path(args.gen), args.k)
        if args.out:
            Path(args.out).write_text(json.dumps(report, indent=1))
        for name, v in report.items():
            print(f"{name:14s} {v}")


if __name__ == "__main__":
    main()
