#!/usr/bin/env python
"""Hook-path smoke test: capture shapes, steering effect, decode speed (run in HyperQwen)."""
from __future__ import annotations

import argparse
import time

import offline
from offline import build_llm, capture_last, find_module, installed_steerer, render, sampling_params


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/app/models/Qwen3.8-27B-W4A16-AutoRound")
    parser.add_argument("--layer", type=int, default=40)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()

    llm, model, tokenizer = build_llm(
        args.model, args.max_model_len, args.gpu_memory_utilization, eager=args.eager, max_num_seqs=args.batch,
        steer_layer=args.layer,
    )
    name, module = find_module(model, args.layer, "down_proj")
    print("module:", name, type(module).__name__, flush=True)
    messages = [{"role": "user", "content": "In one sentence: why does ice float on water?"}]
    captured = capture_last(llm, tokenizer, messages, {args.layer: module})
    vec = captured[args.layer]
    print("captured:", tuple(vec.shape), "norm", float(vec.norm()), flush=True)

    text = render(tokenizer, messages)
    params = sampling_params(96)

    def run(label):
        started = time.time()
        out = llm.generate([text], params, use_tqdm=False)[0].outputs[0]
        seconds = time.time() - started
        print(f"{label}: {len(out.token_ids)} tok {len(out.token_ids) / seconds:.1f} tok/s :: {out.text[:90]!r}", flush=True)
        return out.text

    steerer = installed_steerer()
    base = run("control")
    again = run("control-again")
    import torch

    big = torch.randn(vec.shape[0])
    big = big / big.norm()
    steerer.set(big, alpha=3.0, ref_norm=float(vec.norm()))
    steered = run("alpha=3 random")
    steerer.clear()
    cleared = run("cleared")
    print("deterministic:", base == again, "| steering changes output:", base != steered, "| clear restores:", cleared == base)

    questions = ["why does ice float on water?", "why is the sky blue?", "why do we have seasons?", "why does bread rise?"]
    texts = [render(tokenizer, [{"role": "user", "content": f"In one sentence: {q}"}]) for q in questions[: args.batch]]

    def run_batch(label):
        started = time.time()
        outs = [o.outputs[0] for o in llm.generate(texts, params, use_tqdm=False)]
        tokens = sum(len(o.token_ids) for o in outs)
        print(f"{label}: {len(outs)} seqs {tokens} tok {tokens / (time.time() - started):.1f} tok/s", flush=True)
        return [o.text for o in outs]

    seen = []
    original = offline._last_rows

    def spy(num_rows):
        rows = original(num_rows)
        seen.append((num_rows, None if rows is None else (rows if isinstance(rows, int) else rows.tolist())))
        return rows

    batch_base = run_batch("batch control")
    batch_again = run_batch("batch control-again")
    offline._last_rows = spy
    steerer.set(big, alpha=3.0, ref_norm=float(vec.norm()))
    batch_steered = run_batch("batch alpha=3 random")
    offline._last_rows = original
    steerer.clear()
    prefill_calls = [entry for entry in seen if entry[1] is not None]
    print("prefill row selections:", prefill_calls[:3], "| graph/all-row calls:", sum(e[1] is None for e in seen))
    print("batch deterministic:", batch_base == batch_again,
          "| batch steering changes every output:", all(a != b for a, b in zip(batch_base, batch_steered)),
          "| batch first == single control:", batch_base[0] == base)
    required = [base == again, base != steered, cleared == base, batch_base == batch_again,
                all(a != b for a, b in zip(batch_base, batch_steered)), bool(prefill_calls)]
    if not all(required):
        raise SystemExit(f"smoke checks failed: {required}")
    print("SMOKE OK", flush=True)


if __name__ == "__main__":
    main()
