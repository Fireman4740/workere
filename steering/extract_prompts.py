#!/usr/bin/env python
"""Extract the exact baseline prompts from Qwen inference traces.

The baseline run stored LangChain messages for every successful pass. The last
AIMessage is the model answer; everything before it is the prompt the API saw.
Keeping those prompts fixed makes the offline steering comparison controlled.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from steering.common import read_json, write_jsonl


ROLE = {
    "SystemMessage": "system",
    "HumanMessage": "user",
    "AIMessage": "assistant",
    "ToolMessage": "tool",
}


def message_role(message: dict) -> str:
    identifier = message.get("id") or []
    name = identifier[-1] if identifier else ""
    if name not in ROLE:
        raise ValueError(f"unsupported message type: {name!r}")
    return ROLE[name]


def split_trace(row: dict) -> tuple[list[dict], str | None]:
    """(prompt messages, final assistant answer or None)."""
    messages = row["messages"]
    answer = None
    if messages and message_role(messages[-1]) == "assistant":
        answer = messages[-1]["kwargs"]["content"]
        messages = messages[:-1]
    prompt = [
        {"role": message_role(message), "content": message["kwargs"]["content"]}
        for message in messages
    ]
    return prompt, answer


def prompt_from_trace(row: dict) -> list[dict]:
    return split_trace(row)[0]


def wanted(doc_id: str, ids: set[str], mode: str) -> bool:
    return doc_id in ids if mode == "eval" else doc_id not in ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", action="append", required=True)
    parser.add_argument("--ids", required=True)
    parser.add_argument(
        "--mode",
        choices=["eval", "train"],
        default="eval",
        help="eval: documents listed in --ids; train: every other document (self-contrast)",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--coverage", required=True)
    args = parser.parse_args()

    ids = {row["id"] for row in read_json(args.ids)}
    prompts = {}
    rejected = Counter()
    for trace in args.trace:
        for line in Path(trace).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("role") != "inference" or not wanted(row.get("doc_id", ""), ids, args.mode):
                continue
            if not row.get("ok") or row.get("finish_reason") != "stop":
                rejected[row.get("finish_reason") or row.get("error") or "not_ok"] += 1
                continue
            key = (row["doc_id"], int(row["pass"]))
            if key in prompts:
                continue
            messages, answer = split_trace(row)
            prompts[key] = {
                "doc_id": row["doc_id"],
                "pass": key[1],
                "messages": messages,
                "answer": answer,
                "source_trace": str(trace),
                "resumed_from_run": row.get("resumed_from_run"),
                "prompt_sha256": hashlib.sha256(
                    json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()
                ).hexdigest(),
            }

    rows = [prompts[key] for key in sorted(prompts)]
    coverage = Counter(row["doc_id"] for row in rows)
    summary = {
        "documents": len(coverage),
        "passes": len(rows),
        "passes_per_document": dict(sorted(coverage.items())),
        "documents_with_7_passes": sum(count == 7 for count in coverage.values()),
        "missing_documents": sorted(ids - set(coverage)) if args.mode == "eval" else [],
        "leaked_eval_documents": sorted(ids & set(coverage)) if args.mode == "train" else [],
        "partial_documents": {
            doc_id: count for doc_id, count in sorted(coverage.items()) if count != 7
        },
        "rejected_rows": dict(rejected),
    }
    write_jsonl(args.out, rows)
    Path(args.coverage).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
