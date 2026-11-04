#!/usr/bin/env python
"""E2: early exit by free continuations (BAEE-style) on the CTB prompts.

The greedy trace of each document is grown in chunks of --chunk tokens until it ends; that
full trace is the "control" condition. At every checkpoint, while a document is still
undecided, --k free continuations (temperature --temperature) are sampled from the current
prefix and allowed to finish their thought and answer within --probe-tokens. When at least
two agree on the set of causal ordered pairs and that majority equals the previous
checkpoint's majority, the document is decided: the "early_exit" condition answers with
one of the agreeing continuations. Undecided documents fall back to the full trace.
"""
from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path

from eci.parse import ParseError, parse_triples
from offline import build_llm, read_jsonl, render, sampling_params
from run_vllm import request_params


def answer_key(text: str) -> frozenset | None:
    if "</think>" not in text:
        return None
    try:
        triples = parse_triples(text)
    except (ParseError, ValueError):
        return None
    return frozenset((a, b) for a, label, b in triples if label != "norel")


def majority(texts: list[str]) -> tuple[frozenset | None, str | None]:
    keys = [answer_key(text) for text in texts]
    counts = collections.Counter(key for key in keys if key is not None)
    if not counts:
        return None, None
    best, count = counts.most_common(1)[0]
    if count < 2:
        return None, None
    return best, texts[keys.index(best)]


def write(sink, row, condition, text, tokens, extra) -> None:
    record = {"doc_id": row["doc_id"], "pass": int(row["pass"]), "condition": condition,
              "num_tokens": tokens, "finish_reason": "stop", "text": text, **extra}
    sink.write(json.dumps(record, ensure_ascii=False) + "\n")
    sink.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--doc-ids", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--chunk", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=28672)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--probe-tokens", type=int, default=3072)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--model", default="/app/models/Qwen3.8-27B-W4A16-AutoRound")
    args = parser.parse_args()

    keep = [d.strip() for d in Path(args.doc_ids).read_text().splitlines() if d.strip()]
    rows = [r for r in read_jsonl(args.prompts) if int(r["pass"]) == 0 and r["doc_id"] in set(keep)]
    llm, _, tokenizer = build_llm(args.model, args.max_model_len, args.gpu_memory_utilization,
                                  max_num_seqs=args.batch)
    prompts = {r["doc_id"]: render(tokenizer, r["messages"]) for r in rows}
    state = {r["doc_id"]: {"row": r, "trace": "", "tokens": 0, "ended": False, "decided": False,
                           "previous": None, "checkpoints": 0} for r in rows}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        for row in read_jsonl(out):
            s = state.get(row["doc_id"])
            if s is None:
                continue
            if row["condition"] == "control":
                s["ended"] = True
            elif row["condition"] == "early_exit":
                s["decided"] = True
        print(f"resume: {sum(s['ended'] for s in state.values())} traces already ended, "
              f"{sum(s['decided'] for s in state.values())} decisions already written", flush=True)
    started = time.time()
    with out.open("a") as sink:
        while any(not s["ended"] for s in state.values()):
            growing = [d for d, s in state.items() if not s["ended"]]
            room = {d: args.max_model_len - 8 - len(tokenizer(prompts[d] + state[d]["trace"],
                                                                add_special_tokens=False)["input_ids"])
                    for d in growing}
            outputs = llm.generate([prompts[d] + state[d]["trace"] for d in growing],
                                   [sampling_params(max(1, min(args.chunk, room[d]))) for d in growing],
                                   use_tqdm=False)
            for doc_id, output in zip(growing, outputs):
                s, completion = state[doc_id], output.outputs[0]
                s["trace"] += completion.text
                s["tokens"] += len(completion.token_ids)
                full = room[doc_id] - len(completion.token_ids) < args.chunk
                if completion.finish_reason == "stop" or s["tokens"] >= args.max_tokens or full:
                    s["ended"] = True
                    write(sink, s["row"], "control", s["trace"], s["tokens"],
                          {"truncated": completion.finish_reason != "stop"})
                    if not s["decided"]:
                        s["decided"] = True
                        write(sink, s["row"], "early_exit", s["trace"], s["tokens"],
                              {"exit": "natural_end", "checkpoints": s["checkpoints"]})
            probing = [d for d, s in state.items() if not s["decided"] and not s["ended"]
                       and room[d] - args.chunk > args.probe_tokens]
            if probing:
                params = []
                for d in probing:
                    p = request_params({"temperature": args.temperature, "seed": state[d]["checkpoints"]}, d,
                                       args.probe_tokens)
                    p.n = args.k
                    params.append(p)
                probes = llm.generate([prompts[d] + state[d]["trace"] for d in probing], params, use_tqdm=False)
                for doc_id, probe in zip(probing, probes):
                    s = state[doc_id]
                    s["checkpoints"] += 1
                    key, text = majority([o.text for o in probe.outputs])
                    if key is not None and key == s["previous"]:
                        s["decided"] = True
                        write(sink, s["row"], "early_exit", s["trace"] + text, s["tokens"],
                              {"exit": "agreement", "checkpoints": s["checkpoints"],
                               "exit_trace_tokens": s["tokens"]})
                    s["previous"] = key
            ended = sum(s["ended"] for s in state.values())
            decided = sum(s["decided"] for s in state.values())
            print(f"  t={time.time() - started:.0f}s traces ended {ended}/{len(state)}, "
                  f"early decisions {decided}/{len(state)}", flush=True)
    print(f"done in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
