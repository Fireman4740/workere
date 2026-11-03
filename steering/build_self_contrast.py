#!/usr/bin/env python
"""Build gold-vs-own-error pairs for the self-contrast direction (CPU only).

Input: training-document prompts from extract_prompts.py --mode train (these carry the
model's own answer) and the CTB document table with gold triples. Evaluation documents
are refused, so no gold label of the N13 set reaches the direction.

For every training document the pass with the lowest F1 is kept when it differs from
gold: good_answer is the gold answer in the model's own JSON format, bad_answer is what
the model actually wrote. The contrast is therefore on the answer tokens, not the prompt.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eci.parse import ParseError, parse_triples
from steering.common import (
    binary_counts,
    prf,
    read_json,
    read_jsonl,
    triples_to_map,
    write_jsonl,
)


def render_answer(triples) -> str:
    return json.dumps(
        [{"pair": f"{src},{tgt}", "label": label} for src, label, tgt in triples],
        ensure_ascii=False,
    )


def pass_f1(answer: str, gold_map: dict) -> float | None:
    try:
        predicted = triples_to_map(parse_triples(answer))
    except (ParseError, ValueError):
        return None
    counts = binary_counts(predicted, gold_map)
    return prf(counts["tp"], counts["fp"], counts["fn"])[2]


def build_pairs(prompt_rows, gold_by_doc, eval_ids, max_pairs=0):
    leaked = sorted({row["doc_id"] for row in prompt_rows} & set(eval_ids))
    if leaked:
        raise ValueError(f"evaluation documents in self-contrast input: {leaked[:5]}")
    worst: dict[str, dict] = {}
    for row in prompt_rows:
        doc_id = row["doc_id"]
        gold = gold_by_doc.get(doc_id)
        answer = row.get("answer")
        if not gold or not answer:
            continue
        gold_map = triples_to_map(gold)
        f1 = pass_f1(answer, gold_map)
        if f1 is None or f1 >= 1.0:
            continue
        current = worst.get(doc_id)
        if current is None or f1 < current["bad_f1"]:
            worst[doc_id] = {
                "doc_id": doc_id,
                "pass": row["pass"],
                "messages": row["messages"],
                "good_answer": render_answer(gold),
                "bad_answer": answer,
                "bad_f1": f1,
            }
    pairs = [worst[doc_id] for doc_id in sorted(worst)]
    return pairs[:max_pairs] if max_pairs else pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", required=True, help="train prompts.jsonl with answers")
    parser.add_argument("--docs", required=True, help="tmp/qwen-baseline/docs.jsonl")
    parser.add_argument("--ids", required=True, help="N13 evaluation ids (excluded)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-pairs", type=int, default=0)
    args = parser.parse_args()

    eval_ids = {row["id"] for row in read_json(args.ids)}
    gold_by_doc = {
        row["id"]: [tuple(triple) for triple in row["gold_triples"]]
        for row in read_jsonl(args.docs)
    }
    pairs = build_pairs(read_jsonl(args.prompts), gold_by_doc, eval_ids, args.max_pairs)
    write_jsonl(args.out, pairs)
    mean_f1 = sum(p["bad_f1"] for p in pairs) / len(pairs) if pairs else 0.0
    print(json.dumps({
        "pairs": len(pairs),
        "mean_bad_f1": round(mean_f1, 4),
        "out": str(Path(args.out)),
    }, indent=1))


if __name__ == "__main__":
    main()
