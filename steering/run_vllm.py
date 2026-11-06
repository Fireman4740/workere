#!/usr/bin/env python
"""Generate the CTB baseline prompts offline, one condition after another.

The model is loaded once; conditions run one after another and every document of a
condition is decoded concurrently. Rows are appended as each document finishes and a
rerun skips what is already in --out.

A condition may set, besides an additive steering direction (direction/alpha):
  temperature, seed     sampling instead of greedy decoding
  budget, answer_tokens thinking budget; a trace still open at the budget is closed with
                        FORCE_CLOSE and the model writes its answer freely afterwards
  projector             {"direction", "layer", "strength"}: projection removal (E3)
  heads                 {"edit", "strength"}: per-head output edit from an .npz (E5)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from interventions import clear_all, install_interventions, installed_heads, installed_projector
from offline import build_llm, installed_steerer, load_direction, read_jsonl, render, sampling_params

FORCE_CLOSE = (
    "\n\nConsidering the limited time by the user, I have to give the solution based on "
    "the thinking directly now.\n</think>\n\n"
)


def request_params(condition: dict, doc_id: str, max_tokens: int):
    temperature = float(condition.get("temperature", 0.0))
    seed = None
    if temperature > 0:
        digest = hashlib.sha256(f"{condition.get('seed', 0)}::{doc_id}".encode()).hexdigest()
        seed = int(digest[:8], 16)
    return sampling_params(max_tokens, temperature=temperature, seed=seed)


def generate_condition(llm, tokenizer, todo, condition, max_tokens, sink) -> None:
    """Decode every document of one condition concurrently; write each row as it finishes."""
    engine = llm.llm_engine
    name = condition["name"]
    budget = condition.get("budget")
    answer_tokens = int(condition.get("answer_tokens", 4096))
    pending = {}
    for index, row in enumerate(todo):
        text = render(tokenizer, row["messages"])
        request_id = f"{name}-{index}"
        pending[request_id] = {"row": row, "prompt": text, "phase1": None,
                               "prompt_tokens": len(tokenizer(text, add_special_tokens=False)["input_ids"])}
        engine.add_request(request_id, text, request_params(condition, row["doc_id"], budget or max_tokens))
    started = time.time()
    last_report, finished = started, 0
    while engine.has_unfinished_requests():
        for output in engine.step():
            if not output.finished:
                continue
            base_id = output.request_id.removesuffix("-close")
            state = pending[base_id]
            completion = output.outputs[0]
            if budget and state["phase1"] is None and completion.finish_reason == "length" \
                    and "</think>" not in completion.text:
                state["phase1"] = completion
                engine.add_request(base_id + "-close", state["prompt"] + completion.text + FORCE_CLOSE,
                                   request_params(condition, state["row"]["doc_id"], answer_tokens))
                continue
            pending.pop(base_id)
            finished += 1
            first = state["phase1"]
            text = completion.text if first is None else first.text + FORCE_CLOSE + completion.text
            num_tokens = len(completion.token_ids) + (len(first.token_ids) if first is not None else 0)
            seconds = time.time() - started
            row = state["row"]
            record = {
                "doc_id": row["doc_id"],
                "pass": int(row["pass"]),
                "condition": name,
                **{k: condition.get(k) for k in ("direction", "alpha", "layer", "module", "positions",
                                                 "temperature", "budget", "projector", "heads")},
                "prompt_tokens": state["prompt_tokens"],
                "num_tokens": num_tokens,
                "budget_forced": first is not None,
                "finish_reason": completion.finish_reason,
                "seconds": round(seconds, 3),
                "text": text,
            }
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            sink.flush()
            print(
                f"[{name} {finished}/{len(todo)}] {row['doc_id']} prompt={state['prompt_tokens']} "
                f"out={num_tokens} finish={completion.finish_reason}"
                f"{' forced' if first is not None else ''} at {seconds:.0f}s",
                flush=True,
            )
        if time.time() - last_report > 60:
            last_report = time.time()
            print(f"  {name}: {finished}/{len(todo)} done, {engine.get_num_unfinished_requests()} decoding, "
                  f"{last_report - started:.0f}s elapsed", flush=True)
    print(f"condition={name} finished in {time.time() - started:.0f}s", flush=True)


def load_head_edit(path: str) -> dict[int, tuple]:
    import torch

    data = np.load(path)
    return {int(layer): (torch.from_numpy(data[f"u_{int(layer)}"]).float(),
                         torch.from_numpy(data[f"d_{int(layer)}"]).float())
            for layer in data["layers"]}


def apply_condition(condition: dict, steerer) -> None:
    import torch

    steerer.clear()
    clear_all()
    if condition.get("direction") and condition.get("alpha", 0.0) != 0.0:
        direction, ref_norm = load_direction(condition["direction"])
        steerer.set(direction, condition["alpha"], ref_norm)
    if condition.get("projector"):
        spec = condition["projector"]
        direction = torch.from_numpy(np.load(spec["direction"])["v"]).float()
        installed_projector().set(direction, spec["strength"])
    if condition.get("heads"):
        spec = condition["heads"]
        heads = installed_heads()
        for layer, (u, d) in load_head_edit(spec["edit"]).items():
            heads[layer].set(u, d, spec.get("strength", 1.0))


def install_sites(conditions: list[dict]) -> tuple[int, str]:
    layouts = {(c["layer"], c["module"], c.get("positions", "last")) for c in conditions if c.get("direction")}
    if len(layouts) > 1:
        raise SystemExit(f"one engine serves one steering site; got {sorted(layouts)}")
    layer, module_kind, positions = next(iter(layouts)) if layouts else (40, "down_proj", "last")
    if module_kind != "down_proj":
        raise SystemExit("graph-safe steering is implemented for down_proj only")
    projector_layers = {c["projector"]["layer"] for c in conditions if c.get("projector")}
    if len(projector_layers) > 1:
        raise SystemExit(f"one engine serves one projector layer; got {sorted(projector_layers)}")
    head_layers = set()
    for c in conditions:
        if c.get("heads"):
            head_layers |= {int(x) for x in np.load(c["heads"]["edit"])["layers"]}
    install_interventions(next(iter(projector_layers), None), head_layers)
    return layer, positions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", default="/app/local_extensions/steering/prompts.jsonl")
    parser.add_argument("--conditions", required=True, help="JSON list of conditions")
    parser.add_argument("--only", help="comma-separated condition names to run")
    parser.add_argument("--out", required=True)
    parser.add_argument("--passes", default="0")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--doc-ids", help="file with one doc id per line; only these documents run")
    parser.add_argument("--eager", action="store_true", help="disable CUDA graphs (slower decode)")
    parser.add_argument("--after-condition", help="shell command run after each condition (live scoring)")
    parser.add_argument("--chunked-prefill", action="store_true", help="allow contexts beyond the unchunked-prefill memory limit (no steering)")
    parser.add_argument("--batch", type=int, default=10, help="documents decoded concurrently")
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--max-model-len", type=int, default=49152)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--model", default="/app/models/Qwen3.8-27B-W4A16-AutoRound")
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

    layer, positions = install_sites(conditions)
    llm, _, tokenizer = build_llm(
        args.model, args.max_model_len, args.gpu_memory_utilization, max_num_seqs=args.batch,
        eager=args.eager, steer_layer=layer, steer_positions=positions, chunked_prefill=args.chunked_prefill,
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
        apply_condition(condition, steerer)
        with out_path.open("a") as sink:
            generate_condition(llm, tokenizer, todo, condition, args.max_tokens, sink)
        if args.after_condition:
            print(f"== scores after {name}", flush=True)
            subprocess.run(args.after_condition, shell=True, check=False)
            sys.stdout.flush()


if __name__ == "__main__":
    main()
