"""CausalTimeBank documents and prompts in the exact N13 baseline format.

Evaluation documents keep the stored baseline prompts (prompts.jsonl, pass 0). Training
documents (CTB minus the 50 N13 documents; a demonstration document never demonstrates
itself) get the
same system prompt, one rationale demonstration from another fold, and the same user
template, so traces and probes on them come from the same prompt distribution.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"


def load_suite(path: Path = DATA / "ctb-suite.json") -> dict[str, dict]:
    return {doc["id"]: doc for doc in json.loads(Path(path).read_text())["documents"]}


def load_prompt(path: Path = DATA / "ctb-prompt.json") -> dict:
    return json.loads(Path(path).read_text())


def eval_ids(reference_path: Path = DATA / "reference.json") -> list[str]:
    return list(json.loads(Path(reference_path).read_text())["ids"])


def train_ids(suite: dict[str, dict], prompt: dict, excluded: set[str]) -> list[str]:
    return sorted(doc_id for doc_id, doc in suite.items() if doc_id not in excluded and doc["pair_list"])


def messages_for(doc: dict, prompt: dict) -> list[dict]:
    fold = int(doc["idx"]) % 10
    eligible = [demo for demo in prompt["demos"] if demo["fold"] != fold and demo["id"] != doc["id"]]
    demo = random.Random(f"steer-train::{doc['id']}").choice(eligible)
    template = prompt["user_template"]
    return [
        {"role": "system", "content": prompt["system"]},
        {"role": "user", "content": template.format(doc_text=demo["text"])},
        {"role": "assistant", "content": demo["answer"]},
        {"role": "user", "content": template.format(doc_text=doc["text"])},
    ]


def unordered_pairs(doc: dict) -> list[tuple[str, str]]:
    return sorted({tuple(sorted(pair)) for pair in doc["pair_list"]})


def causal_pairs(doc: dict) -> set[tuple[str, str]]:
    return {tuple(sorted((a, b))) for a, label, b in doc["gold"] if label != "norel"}


def surfaces(doc: dict) -> dict[str, str]:
    return dict(doc["surfaces"])


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Write N13-format prompts for the CTB training documents.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    suite, prompt = load_suite(), load_prompt()
    ids = train_ids(suite, prompt, set(eval_ids()))
    rows = [{"doc_id": doc_id, "pass": 0, "messages": messages_for(suite[doc_id], prompt)} for doc_id in ids]
    Path(args.out).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    print(f"{len(rows)} training prompts -> {args.out}")


if __name__ == "__main__":
    main()
