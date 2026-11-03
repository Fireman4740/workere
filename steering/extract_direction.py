#!/usr/bin/env python
"""Extract contrastive activation directions, several layers per prefill.

Methods (run in this order for the pilot):
  hint  last prompt token; system message + causal / no-causal annotation hint
  demo  last prompt token; demo answer kept / replaced by an all-norel answer
  self  answer tokens; gold answer vs the model's own wrong answer, on training
        documents that are NOT in the evaluation set (see build_self_contrast.py)

One unit-norm direction per layer is written to <out-dir>/<method>_L<layer>_<module>.npz
together with ref_norm, the median module-output norm, which the generation runner uses
to turn the dimensionless alpha into an absolute dose.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from offline import (
    build_llm,
    capture_answer_span,
    capture_last,
    find_module,
    parse_layers,
    read_jsonl,
    render,
)

HINT_POSITIVE = "\n\nAnnotation hint: this document contains at least one causal relation between the marked events."
HINT_NEGATIVE = "\n\nAnnotation hint: this document contains no causal relation between the marked events."
DEMO_NEGATIVE = "No explicit causal marker connects the marked events.\n\n[]"


def with_hint(messages: list[dict], positive: bool) -> list[dict]:
    out = [dict(message) for message in messages]
    out[0]["content"] += HINT_POSITIVE if positive else HINT_NEGATIVE
    return out


def with_demo(messages: list[dict], positive: bool) -> list[dict]:
    out = [dict(message) for message in messages]
    if positive:
        return out
    for index, message in enumerate(out):
        if message["role"] == "assistant":
            out[index]["content"] = DEMO_NEGATIVE
            return out
    raise ValueError("prompt has no assistant demo message")


def unit_direction(diffs: np.ndarray) -> tuple[np.ndarray, dict]:
    mean = diffs.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm == 0:
        raise RuntimeError("mean direction is zero")
    vector = mean / norm
    projections = diffs @ vector
    cosines = projections / np.linalg.norm(diffs, axis=1)
    d_prime = float(projections.mean() / projections.std()) if projections.std() > 0 else float("inf")
    stats = {
        "mean_diff_norm": norm,
        "mean_projection": float(projections.mean()),
        "separation_d": d_prime,
        "mean_cosine": float(cosines.mean()),
        "min_cosine": float(cosines.min()),
        "max_cosine": float(cosines.max()),
    }
    return vector.astype(np.float32), stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["hint", "demo", "self"], required=True)
    parser.add_argument("--prompts", help="prompts.jsonl (hint/demo) or self_pairs.jsonl (self)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--pass", dest="pass_index", type=int, default=0)
    parser.add_argument("--layers", default="24,32,40,48")
    parser.add_argument("--module", choices=["down_proj", "o_proj"], default="down_proj")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=20480)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--model", default="/app/models/Qwen3.8-27B-W4A16-AutoRound")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    layers = parse_layers(args.layers)
    rows = read_jsonl(args.prompts)
    if args.method != "self":
        rows = [row for row in rows if int(row["pass"]) == args.pass_index]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("no input rows")

    if args.dry_run:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        lengths = []
        for row in rows:
            if args.method == "self":
                lengths.append(max(
                    len(tokenizer(render(tokenizer, row["messages"]) + row[key], add_special_tokens=False)["input_ids"])
                    for key in ("good_answer", "bad_answer")
                ))
            else:
                build = with_hint if args.method == "hint" else with_demo
                lengths.append(max(
                    len(tokenizer(render(tokenizer, build(row["messages"], flag)), add_special_tokens=False)["input_ids"])
                    for flag in (True, False)
                ))
        print(json.dumps({
            "method": args.method,
            "rows": len(rows),
            "max_tokens": max(lengths),
            "over_max_model_len": sum(length >= args.max_model_len for length in lengths),
        }, indent=1))
        return

    llm, model, tokenizer = build_llm(
        args.model, args.max_model_len, args.gpu_memory_utilization, max_num_seqs=1, eager=True
    )
    modules = {layer: find_module(model, layer, args.module)[1] for layer in layers}

    diffs: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    norms: dict[int, list[float]] = {layer: [] for layer in layers}
    docs = []
    for index, row in enumerate(rows, 1):
        if args.method == "self":
            pos = capture_answer_span(llm, tokenizer, row["messages"], row["good_answer"], modules)
            neg = capture_answer_span(llm, tokenizer, row["messages"], row["bad_answer"], modules)
        else:
            build = with_hint if args.method == "hint" else with_demo
            pos = capture_last(llm, tokenizer, build(row["messages"], True), modules)
            neg = capture_last(llm, tokenizer, build(row["messages"], False), modules)
        for layer in layers:
            diffs[layer].append((pos[layer] - neg[layer]).numpy())
            norms[layer].extend([float(pos[layer].norm()), float(neg[layer].norm())])
        docs.append(row.get("doc_id", str(index)))
        print(f"[{index}/{len(rows)}] {docs[-1]}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for layer in layers:
        matrix = np.stack(diffs[layer]).astype(np.float32)
        vector, stats = unit_direction(matrix)
        ref_norm = float(np.median(norms[layer]))
        path = out_dir / f"{args.method}_L{layer}_{args.module}.npz"
        np.savez(
            path,
            v=vector,
            diffs=matrix,
            docs=np.array(docs),
            ref_norm=np.array(ref_norm),
            method=np.array(args.method),
            layer=np.array(layer),
            module=np.array(args.module),
        )
        summary[str(layer)] = {"file": str(path), "ref_norm": ref_norm, **stats}
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
