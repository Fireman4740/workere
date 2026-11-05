#!/usr/bin/env python
"""E3/E5: hesitation directions from the model's own traces on training documents.

Sentences of a thinking trace are labelled hesitant (they reopen a decision: "But",
"Wait", "Should we include", ...) or decisive (everything else, at least four words).
Activations are read at the first four tokens of each sentence.

E3 (Reflection Steering recipe): per down_proj layer, raw = mean(hesitant) - mean(decisive),
projected on the top principal components of all sentence activations, made orthogonal
to the mean activation (general reasoning), normalised. The layer with the best held-out
separation is used for projection removal.
E5 (ThinkEdit recipe): per full-attention layer, d_L = unit(mean_h - mean_d) of o_proj
outputs; for each head, its contribution along d_L is (x_h . u_h) with u = W_o^T d_L;
the heads whose contribution most separates hesitant from decisive sentences are edited.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import numpy as np

from interventions import capture_rows
from offline import build_llm, find_module, read_jsonl, render

HESITANT = re.compile(
    r"^(but|wait|however|actually|hmm|alternatively|maybe|perhaps|hold on|on second thought|"
    r"let me (re|double)|let's (re|double)|should we|should i|reconsider|re-?check|double-?check|"
    r"again|or maybe|is that|is it|does that|would that)\b",
    re.IGNORECASE,
)
SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
ONSET = 4
DOWN_LAYERS = (16, 24, 32, 40, 48, 56)
ATTN_LAYERS = tuple(range(3, 64, 4))
HEAD_DIM = 256


def sentences(thinking: str) -> list[tuple[int, str]]:
    out, cursor = [], 0
    for part in SPLIT.split(thinking):
        start = thinking.find(part, cursor)
        cursor = start + len(part)
        if part.strip():
            out.append((start + len(part) - len(part.lstrip()), part.strip()))
    return out


def label(sentence: str) -> str | None:
    if HESITANT.match(sentence):
        return "h"
    return "d" if len(sentence.split()) >= 4 else None


def sentence_windows(tokenizer, prompt: str, thinking: str, per_class: int, seed: str,
                     max_tokens: int) -> tuple[str, list[tuple[str, list[int]]]]:
    full = prompt + thinking
    encoded = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    starts = [a for a, _ in encoded["offset_mapping"]]
    if len(starts) > max_tokens:
        return "", []
    labelled = [(lab, len(prompt) + start) for start, text in sentences(thinking)
                if (lab := label(text)) is not None]
    rng = random.Random(seed)
    chosen = []
    for cls in ("h", "d"):
        items = [pos for lab, pos in labelled if lab == cls]
        chosen += [(cls, pos) for pos in rng.sample(items, min(per_class, len(items)))]
    windows, index = [], 0
    for cls, char in sorted(chosen, key=lambda item: item[1]):
        while index < len(starts) and starts[index] < char:
            index += 1
        if index + ONSET <= len(starts):
            windows.append((cls, list(range(index, index + ONSET))))
    return full, windows


def collect(llm, tokenizer, model, items, taps_spec, per_class, max_tokens):
    taps = {}
    for name, (layer, kind, side) in taps_spec.items():
        taps[name] = (find_module(model, layer, kind)[1], side)
    rows = {name: [] for name in taps}
    labels, docs = [], []
    for number, (doc_id, prompt, thinking) in enumerate(items, 1):
        full, windows = sentence_windows(tokenizer, prompt, thinking, per_class, doc_id, max_tokens)
        if not windows:
            continue
        positions = sorted({p for _, window in windows for p in window})
        where = {p: i for i, p in enumerate(positions)}
        store = capture_rows(llm, full, taps, positions)
        for name, tensor in store.items():
            means = [tensor[[where[p] for p in window]].mean(0).numpy().astype(np.float16) for _, window in windows]
            rows[name].append(np.stack(means))
        labels += [cls for cls, _ in windows]
        docs += [doc_id] * len(windows)
        print(f"[{number}/{len(items)}] {doc_id}: {sum(c == 'h' for c, _ in windows)} hesitant, "
              f"{sum(c == 'd' for c, _ in windows)} decisive", flush=True)
    return {name: np.concatenate(parts) for name, parts in rows.items()}, np.array(labels), np.array(docs)


def denoised_direction(x: np.ndarray, hes: np.ndarray, rank: int) -> np.ndarray:
    raw = x[hes].mean(0) - x[~hes].mean(0)
    centred = x - x.mean(0)
    sample = centred[np.random.default_rng(0).choice(len(centred), min(len(centred), 8000), replace=False)]
    _, _, vt = np.linalg.svd(sample, full_matrices=False)
    basis = vt[:rank]
    v = basis.T @ (basis @ raw)
    mu = x.mean(0)
    mu = mu / np.linalg.norm(mu)
    v = v - (v @ mu) * mu
    return v / np.linalg.norm(v)


def separation(x: np.ndarray, hes: np.ndarray, v: np.ndarray) -> float:
    p = x @ v
    a, b = p[hes], p[~hes]
    return float((a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2 + 1e-12))


def e3_directions(vectors, labels, docs, out_dir: Path, rank: int) -> dict:
    ids = sorted(set(docs))
    held = set(random.Random("e3-split").sample(ids, max(1, len(ids) // 5)))
    test = np.array([d in held for d in docs])
    hes = labels == "h"
    summary = {}
    for layer in DOWN_LAYERS:
        x = vectors[f"down{layer}"].astype(np.float32)
        v = denoised_direction(x[~test], hes[~test], rank)
        ref_norm = float(np.median(np.linalg.norm(x, axis=1)))
        sep = separation(x[test], hes[test], v)
        np.savez(out_dir / f"refl_L{layer}.npz", v=v.astype(np.float32), layer=layer, ref_norm=ref_norm,
                 heldout_separation=sep)
        summary[layer] = {"heldout_separation": round(sep, 3), "ref_norm": round(ref_norm, 3)}
    best = max(summary, key=lambda layer: summary[layer]["heldout_separation"])
    rand = np.random.default_rng(20261104).standard_normal(vectors[f"down{best}"].shape[1])
    np.savez(out_dir / f"refl_rand_L{best}.npz", v=(rand / np.linalg.norm(rand)).astype(np.float32), layer=best)
    (out_dir / "e3_best_layer.txt").write_text(f"{best}\n")
    return {"layers": summary, "best": best}


def o_proj_u(module, direction: np.ndarray) -> np.ndarray:
    import torch

    with torch.inference_mode():
        eye = torch.eye(module.input_size, dtype=torch.bfloat16, device="cuda")
        out = module(eye)
        out = out[0] if isinstance(out, tuple) else out
        d = torch.from_numpy(direction).to(out)
        return (out.float() @ d.float()).cpu().numpy()


def e5_edit(llm, tokenizer, model, items, out_vectors, labels, out_dir: Path, per_class: int,
            max_tokens: int, fraction: float) -> dict:
    hes = labels == "h"
    directions, us = {}, {}
    for layer in ATTN_LAYERS:
        x = out_vectors[f"oout{layer}"].astype(np.float32)
        d = x[hes].mean(0) - x[~hes].mean(0)
        directions[layer] = d / np.linalg.norm(d)
        us[layer] = o_proj_u(find_module(model, layer, "o_proj")[1], directions[layer])
    taps = {f"oin{layer}": (layer, "o_proj", "in") for layer in ATTN_LAYERS}
    inputs, head_labels, _ = collect(llm, tokenizer, model, items, taps, per_class, max_tokens)
    head_hes = head_labels == "h"
    scores = []
    for layer in ATTN_LAYERS:
        x = inputs[f"oin{layer}"].astype(np.float32)
        u = us[layer]
        for head in range(len(u) // HEAD_DIM):
            sl = slice(head * HEAD_DIM, (head + 1) * HEAD_DIM)
            contribution = x[:, sl] @ u[sl]
            scores.append((float(contribution[head_hes].mean() - contribution[~head_hes].mean()), layer, head))
    scores.sort(reverse=True)
    chosen = scores[: max(1, round(fraction * len(scores)))]
    payload, layers = {}, sorted({layer for _, layer, _ in chosen})
    for layer in layers:
        mask = np.zeros_like(us[layer])
        for _, chosen_layer, head in chosen:
            if chosen_layer == layer:
                mask[head * HEAD_DIM:(head + 1) * HEAD_DIM] = 1.0
        payload[f"u_{layer}"] = (us[layer] * mask).astype(np.float32)
        payload[f"d_{layer}"] = directions[layer].astype(np.float32)
    np.savez(out_dir / "heads_edit.npz", layers=np.array(layers), **payload)
    rng = np.random.default_rng(20261104)
    random_heads = rng.choice(len(scores), len(chosen), replace=False)
    rand_payload, rand_layers = {}, set()
    for index in random_heads:
        _, layer, head = sorted(scores, key=lambda s: (s[1], s[2]))[index]
        rand_layers.add(layer)
        mask = rand_payload.get(f"m_{layer}", np.zeros_like(us[layer]))
        mask[head * HEAD_DIM:(head + 1) * HEAD_DIM] = 1.0
        rand_payload[f"m_{layer}"] = mask
    np.savez(out_dir / "heads_edit_random.npz", layers=np.array(sorted(rand_layers)),
             **{f"u_{layer}": (us[layer] * rand_payload[f"m_{layer}"]).astype(np.float32) for layer in rand_layers},
             **{f"d_{layer}": directions[layer].astype(np.float32) for layer in rand_layers})
    return {"heads": [{"layer": layer, "head": head, "score": round(score, 4)} for score, layer, head in chosen]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["e3", "e5"], required=True)
    parser.add_argument("--prompts", required=True, help="training prompts jsonl (doc_id, pass, messages)")
    parser.add_argument("--traces", required=True, help="greedy generations on those prompts")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--per-class", type=int, default=60)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--head-fraction", type=float, default=0.04)
    parser.add_argument("--e5-docs", type=int, default=40)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--model", default="/app/models/Qwen3.8-27B-W4A16-AutoRound")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    llm, model, tokenizer = build_llm(args.model, args.max_model_len, args.gpu_memory_utilization, eager=True)
    messages = {r["doc_id"]: r["messages"] for r in read_jsonl(args.prompts)}
    items = []
    for trace in read_jsonl(args.traces):
        if "</think>" in trace["text"] and trace["doc_id"] in messages:
            items.append((trace["doc_id"], render(tokenizer, messages[trace["doc_id"]]),
                          trace["text"].split("</think>")[0]))
    limit = args.max_model_len - 8
    if args.stage == "e3":
        taps = {f"down{layer}": (layer, "down_proj", "out") for layer in DOWN_LAYERS}
        vectors, labels, docs = collect(llm, tokenizer, model, items, taps, args.per_class, limit)
        summary = e3_directions(vectors, labels, docs, out_dir, args.rank)
        summary["sentences"] = {"hesitant": int((labels == "h").sum()), "decisive": int((labels == "d").sum())}
        (out_dir / "e3_summary.json").write_text(json.dumps(summary, indent=1))
    else:
        cache = out_dir / "e5_oproj_outputs.npz"
        if cache.exists():
            data = np.load(cache)
            out_vectors = {k: data[k] for k in data.files if k.startswith("oout")}
            labels = data["labels"]
        else:
            taps = {f"oout{layer}": (layer, "o_proj", "out") for layer in ATTN_LAYERS}
            out_vectors, labels, _ = collect(llm, tokenizer, model, items, taps, args.per_class, limit)
            np.savez(cache, labels=labels, **out_vectors)
        subset = random.Random("e5-subset").sample(items, min(args.e5_docs, len(items)))
        summary = e5_edit(llm, tokenizer, model, subset, out_vectors, labels, out_dir, args.per_class,
                          limit, args.head_fraction)
        (out_dir / "e5_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
