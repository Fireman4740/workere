#!/usr/bin/env python
"""Tokenize the recovered baseline prompts with the served checkpoint's tokenizer.

This is a CPU-only dry run. It checks the chat-template flags, the prompt token
distribution, and whether the demo CoT survives the template.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def render(tokenizer, messages, mode):
    if mode == "kwargs":
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": True, "preserve_thinking": True},
        )
    if mode == "direct":
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
            preserve_thinking=True,
        )
    if mode == "enable":
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
    if mode == "plain":
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    raise ValueError(mode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/app/models/Qwen3.8-27B-W4A16-AutoRound")
    parser.add_argument("--prompts", default="/app/local_extensions/steering/prompts.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--mode", default="kwargs", choices=["kwargs", "direct", "enable", "plain"])
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    rows = [json.loads(line) for line in Path(args.prompts).read_text().splitlines() if line.strip()]
    if args.limit:
        rows = rows[: args.limit]

    lengths = []
    errors = []
    for row in rows:
        try:
            text = render(tokenizer, row["messages"], args.mode)
            lengths.append(len(tokenizer(text, add_special_tokens=False)["input_ids"]))
        except Exception as error:  # noqa: BLE001
            errors.append({"doc_id": row["doc_id"], "pass": row["pass"], "error": f"{type(error).__name__}: {error}"})
    lengths.sort()
    print(json.dumps({
        "mode": args.mode,
        "prompts": len(rows),
        "rendered": len(lengths),
        "errors": errors[:5],
        "error_count": len(errors),
        "min_tokens": lengths[0] if lengths else None,
        "median_tokens": lengths[len(lengths) // 2] if lengths else None,
        "p95_tokens": lengths[int(0.95 * (len(lengths) - 1))] if lengths else None,
        "max_tokens": lengths[-1] if lengths else None,
        "over_16384": sum(length > 16384 for length in lengths),
        "over_32768": sum(length > 32768 for length in lengths),
    }, indent=1))

    sample = rows[0]
    text = render(tokenizer, sample["messages"], args.mode)
    print("sample_prefix:", text[:500].replace("\n", "\\n"))
    print("sample_suffix:", text[-800:].replace("\n", "\\n"))


if __name__ == "__main__":
    main()
