#!/usr/bin/env python
"""Generate the CTB50 baseline prompts offline, one steering condition after another.

Run inside the HyperQwen container with the GPU free. The model is loaded once; the
conditions file (see pilot_conditions.py) is processed condition by condition, so an
interrupted run leaves complete conditions behind. Rows are appended as soon as they
finish and a rerun skips what is already in --out.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from offline import (
    build_llm,
    load_direction,
    read_jsonl,
    render,
    sampling_params,
    installed_steerer,
)


def generate_condition(llm, tokenizer, todo, condition, max_tokens, sink) -> None:
    """Decode every document of one condition concurrently; write each row as it finishes."""
    engine = llm.llm_engine
    name = condition["name"]
    pending = {}
    for index, row in enumerate(todo):
        text = render(tokenizer, row["messages"])
        request_id = f"{name}-{index}"
        pending[request_id] = (row, len(tokenizer(text, add_special_tokens=False)["input_ids"]))
        engine.add_request(request_id, text, sampling_params(max_tokens))
    started = time.time()
    last_report, finished = started, 0
    while engine.has_unfinished_requests():
        for output in engine.step():
            if not output.finished:
                continue
            finished += 1
            row, prompt_tokens = pending.pop(output.request_id)
            completion = output.outputs[0]
            seconds = time.time() - started
            num_tokens = len(completion.token_ids)
            record = {
                "doc_id": row["doc_id"],
                "pass": int(row["pass"]),
                "condition": name,
                **{k: condition.get(k) for k in ("direction", "alpha", "layer", "module", "positions")},
                "prompt_tokens": prompt_tokens,
                "num_tokens": num_tokens,
                "finish_reason": completion.finish_reason,
                "seconds": round(seconds, 3),
                "text": completion.text,
            }
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            sink.flush()
            print(
                f"[{name} {finished}/{len(todo)}] {row['doc_id']} prompt={prompt_tokens} "
                f"out={num_tokens} finish={completion.finish_reason} at {seconds:.0f}s",
                flush=True,
            )
        if time.time() - last_report > 60:
            last_report = time.time()
            print(f"  {name}: {finished}/{len(todo)} done, {engine.get_num_unfinished_requests()} decoding, "
                  f"{last_report - started:.0f}s elapsed", flush=True)
    print(f"condition={name} finished in {time.time() - started:.0f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", default="/app/local_extensions/steering/prompts.jsonl")
    parser.add_argument("--conditions", required=True, help="JSON list from pilot_conditions.py")
    parser.add_argument("--only", help="comma-separated condition names to run")
    parser.add_argument("--out", required=True)
    parser.add_argument("--passes", default="0")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--doc-ids", help="file with one doc id per line; only these documents run")
    parser.add_argument("--eager", action="store_true", help="disable CUDA graphs (slower decode)")
    parser.add_argument("--after-condition", help="shell command run after each condition (live scoring)")
    parser.add_argument("--batch", type=int, default=10, help="documents decoded concurrently")
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--max-model-len", type=int, default=49152)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--model", default="/app/models/Qwen3.8-27B-W4A16-AutoRound")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    wanted_passes = {int(value) for value in args.passes.split(",") if value.strip()}
    rows = [row for row in read_jsonl(args.prompts) if int(row["pass"]) in wanted_passes]
    if args.doc_ids:
        keep_docs = [d.strip() for d in Path(args.doc_ids).read_text().splitlines() if d.strip()]
        order = {d: i for i, d in enumerate(keep_docs)}
        rows = sorted((r for r in rows if r["doc_id"] in order), key=lambda r: (order[r["doc_id"]], int(r["pass"])))
    if args.limit:
        rows = rows[: args.limit]
    conditions = json.loads(Path(args.conditions).read_text())
    if args.only:
        keep = set(args.only.split(","))
        conditions = [c for c in conditions if c["name"] in keep]

    if args.dry_run:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        lengths = [len(tokenizer(render(tokenizer, r["messages"]), add_special_tokens=False)["input_ids"]) for r in rows]
        print(json.dumps({
            "conditions": [c["name"] for c in conditions],
            "prompts": len(rows),
            "min_tokens": min(lengths) if lengths else None,
            "max_tokens": max(lengths) if lengths else None,
            "over_max_model_len": sum(n + args.max_tokens > args.max_model_len for n in lengths),
        }, indent=1))
        return

    layouts = {(c["layer"], c["module"], c.get("positions", "last")) for c in conditions if c.get("direction")}
    if len(layouts) > 1:
        raise SystemExit(f"one engine serves one steering site; got {sorted(layouts)}")
    layer, module_kind, positions = next(iter(layouts)) if layouts else (40, "down_proj", "last")
    if module_kind != "down_proj":
        raise SystemExit("graph-safe steering is implemented for down_proj only")
    llm, model, tokenizer = build_llm(
        args.model, args.max_model_len, args.gpu_memory_utilization, max_num_seqs=args.batch,
        eager=args.eager, steer_layer=layer, steer_positions=positions,
    )
    steerer = installed_steerer()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        done = {(r["doc_id"], int(r["pass"]), r["condition"]) for r in read_jsonl(out_path)}

    for condition in conditions:
        name = condition["name"]
        todo = [r for r in rows if (r["doc_id"], int(r["pass"]), name) not in done]
        print(f"condition={name} todo={len(todo)}/{len(rows)}", flush=True)
        steerer.clear()
        if condition.get("direction") and condition["alpha"] != 0.0:
            direction, ref_norm = load_direction(condition["direction"])
            steerer.set(direction, condition["alpha"], ref_norm)
        with out_path.open("a") as sink:
            generate_condition(llm, tokenizer, todo, condition, args.max_tokens, sink)
        if args.after_condition:
            print(f"== scores after {name}", flush=True)
            subprocess.run(args.after_condition, shell=True, check=False)
            sys.stdout.flush()


if __name__ == "__main__":
    main()
