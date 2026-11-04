#!/usr/bin/env python
"""E4: linear probe on the residual stream, one feature vector per candidate pair.

Each document is prefilled once: the N13 prompt, then (variant B) the model's own thinking
trace, then a list of its candidate intra-sentence pairs, one per line ("eiA (surface) /
eiB (surface):"). The residual stream at the last token of each line is the pair's feature.
A logistic regression per layer is trained on the training documents (CTB minus the N13
50 minus the 20 demonstrations); layer, L2 strength and threshold are chosen by
document-grouped cross-validation on those documents only, then applied once to the 47
controlled evaluation documents.

Stage "features" needs the GPU; stage "fit" is CPU-only and reads the saved features.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

import ctb
from common import aggregate_counts, binary_counts, binary_metric, read_json, read_jsonl

LAYERS = (16, 24, 32, 40, 48, 56)
HEADER = "\n\nPair check: for each candidate pair below, is there a causal relation between the two events?\n"


def pair_block(doc: dict) -> tuple[str, list[tuple[str, str]], list[int]]:
    surf = ctb.surfaces(doc)
    pairs = ctb.unordered_pairs(doc)
    text, ends = HEADER, []
    for a, b in pairs:
        text += f"{a} ({surf.get(a, '?')}) / {b} ({surf.get(b, '?')}):"
        ends.append(len(text))
        text += "\n"
    return text, pairs, ends


def end_positions(tokenizer, full: str, char_ends: list[int]) -> list[int]:
    offsets = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    positions, index = [], 0
    for end in char_ends:
        while index < len(offsets) and offsets[index][1] < end:
            index += 1
        positions.append(min(index, len(offsets) - 1))
    return positions


def features(args) -> None:
    from interventions import capture_rows, find_layer
    from offline import build_llm, render

    suite, prompt = ctb.load_suite(), ctb.load_prompt()
    evals = set(ctb.eval_ids())
    eval_messages = {r["doc_id"]: r["messages"] for r in read_jsonl(args.eval_prompts) if int(r["pass"]) == 0}
    thinking = {}
    for path in filter(None, [args.train_traces, args.eval_traces]):
        for row in read_jsonl(path):
            if "</think>" in row["text"]:
                thinking[row["doc_id"]] = row["text"].split("</think>")[0]
    llm, model, tokenizer = build_llm(args.model, args.max_model_len, args.gpu_memory_utilization, eager=True)
    taps = {f"L{layer}": (find_layer(model, layer), "resid") for layer in LAYERS}
    docs = [d for d in ctb.train_ids(suite, prompt, evals)] + [d for d in ctb.eval_ids() if d in eval_messages]
    out = {f"L{layer}": [] for layer in LAYERS}
    meta = []
    for number, doc_id in enumerate(docs, 1):
        doc = suite[doc_id]
        messages = eval_messages[doc_id] if doc_id in evals else ctb.messages_for(doc, prompt)
        if args.variant == "B" and doc_id not in thinking:
            print(f"skip {doc_id}: no finished trace", flush=True)
            continue
        head = render(tokenizer, messages) + (thinking[doc_id] if args.variant == "B" else "")
        block, pairs, ends = pair_block(doc)
        full = head + block
        positions = end_positions(tokenizer, full, [len(head) + e for e in ends])
        if positions[-1] >= args.max_model_len - 2:
            print(f"skip {doc_id}: {positions[-1]} tokens", flush=True)
            continue
        store = capture_rows(llm, full, taps, positions)
        for name, tensor in store.items():
            out[name].append(tensor.numpy().astype(np.float16))
        gold = ctb.causal_pairs(doc)
        meta += [{"doc": doc_id, "a": a, "b": b, "gold": (a, b) in gold, "split": "eval" if doc_id in evals else "train"}
                 for a, b in pairs]
        print(f"[{number}/{len(docs)}] {doc_id} pairs={len(pairs)} tokens={positions[-1] + 1}", flush=True)
    target = Path(args.out_dir)
    target.mkdir(parents=True, exist_ok=True)
    np.savez(target / f"features_{args.variant}.npz", **{k: np.concatenate(v) for k, v in out.items()})
    (target / f"meta_{args.variant}.json").write_text(json.dumps(meta))


def train_logreg(x, y, l2: float, steps: int = 300):
    import torch

    xt, yt = torch.from_numpy(x).float(), torch.from_numpy(y.astype(np.float32))
    w = torch.zeros(x.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    pos = max(float(yt.mean()), 1e-3)
    weight = torch.where(yt > 0, 0.5 / pos, 0.5 / (1 - pos))
    opt = torch.optim.LBFGS([w, b], max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        logits = xt @ w + b
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(logits, yt, weight=weight)
                + l2 * (w @ w) / 2)
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach().numpy(), float(b.detach())


def standardise(train, other):
    mean, std = train.mean(0), train.std(0) + 1e-6
    return (train - mean) / std, (other - mean) / std


def best_threshold(scores, y) -> tuple[float, float]:
    best = (0.0, 0.5)
    for t in np.quantile(scores, np.linspace(0.5, 0.995, 100)):
        pred = scores >= t
        tp, fp, fn = int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        best = max(best, (f1, float(t)))
    return best


def auroc(scores, y) -> float:
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)) if n_pos and n_neg else float("nan")


def doc_counts(meta_rows, predicted, reference) -> dict[str, dict]:
    per_doc = {}
    for doc_id in sorted({m["doc"] for m in meta_rows}):
        pred = {}
        for m, p in zip(meta_rows, predicted):
            if m["doc"] == doc_id and p:
                pred[(m["a"], m["b"])] = "causes"
                pred[(m["b"], m["a"])] = "causedby"
        gold = {tuple(k.split(",", 1)): v for k, v in reference["gold"][doc_id].items()}
        per_doc[doc_id] = binary_counts(pred, gold)
    return per_doc


def fit(args) -> None:
    from compare import paired_delta

    reference = read_json(ctb.DATA / "reference.json")
    data = np.load(Path(args.out_dir) / f"features_{args.variant}.npz")
    meta = json.loads((Path(args.out_dir) / f"meta_{args.variant}.json").read_text())
    split = np.array([m["split"] for m in meta])
    y = np.array([m["gold"] for m in meta])
    docs = np.array([m["doc"] for m in meta])
    train = split == "train"
    eval_meta = [m for m in meta if m["split"] == "eval" and m["doc"] in set(reference["qwen_ids"])]
    eval_mask = np.array([m["split"] == "eval" and m["doc"] in set(reference["qwen_ids"]) for m in meta])
    train_docs = sorted(set(docs[train]))
    folds = {d: i % 5 for i, d in enumerate(random.Random("e4-folds").sample(train_docs, len(train_docs)))}
    fold = np.array([folds.get(d, -1) for d in docs])
    cv = []
    for layer in LAYERS:
        x = data[f"L{layer}"].astype(np.float32)
        for l2 in (1e-3, 1e-2, 1e-1, 1.0):
            scores = np.zeros(len(y))
            for k in range(5):
                tr, te = train & (fold != k), train & (fold == k)
                xtr, xte = standardise(x[tr], x[te])
                w, b = train_logreg(xtr, y[tr], l2)
                scores[te] = xte @ w + b
            f1, threshold = best_threshold(scores[train], y[train])
            cv.append({"layer": layer, "l2": l2, "cv_f1": round(100 * f1, 2),
                       "cv_auroc": round(auroc(scores[train], y[train]), 4), "threshold": threshold})
            print(f"cv layer={layer} l2={l2} F1={100 * f1:.2f} AUROC={cv[-1]['cv_auroc']}", flush=True)
    best = max(cv, key=lambda row: row["cv_f1"])
    x = data[f"L{best['layer']}"].astype(np.float32)
    xtr, xev = standardise(x[train], x[eval_mask])
    w, b = train_logreg(xtr, y[train], best["l2"])
    scores = xev @ w + b
    predicted = scores >= best["threshold"]
    probe_docs = doc_counts(eval_meta, predicted, reference)
    qwen_docs = {d: binary_counts({tuple(k.split(",", 1)): v for k, v in reference["qwen"][d].items()},
                                  {tuple(k.split(",", 1)): v for k, v in reference["gold"][d].items()})
                 for d in probe_docs}
    qwen_pos = [reference["qwen"][m["doc"]].get(f"{m['a']},{m['b']}", "norel") != "norel"
                or reference["qwen"][m["doc"]].get(f"{m['b']},{m['a']}", "norel") != "norel" for m in eval_meta]
    fused = {"probe_or_qwen": predicted | np.array(qwen_pos), "probe_and_qwen": predicted & np.array(qwen_pos)}
    result = {"variant": args.variant, "selected": best, "cv": cv,
              "eval_auroc": round(auroc(scores, y[eval_mask]), 4),
              "probe": binary_metric(aggregate_counts(probe_docs.values())),
              "qwen_7pass_baseline": binary_metric(aggregate_counts(qwen_docs.values()))}
    delta, lo, hi = paired_delta(probe_docs, qwen_docs, 10000)
    result["probe_vs_qwen_dF1"] = [round(delta, 2), round(lo, 2), round(hi, 2)]
    for name, pred in fused.items():
        counts = doc_counts(eval_meta, pred, reference)
        delta, lo, hi = paired_delta(counts, qwen_docs, 10000)
        result[name] = {**binary_metric(aggregate_counts(counts.values())), "vs_qwen_dF1": [round(delta, 2), round(lo, 2), round(hi, 2)]}
    (Path(args.out_dir) / f"result_{args.variant}.json").write_text(json.dumps(result, indent=1))
    for key in ("probe", "qwen_7pass_baseline", "probe_or_qwen", "probe_and_qwen"):
        m = result[key]
        print(f"{key:22s} F1={100 * m['f1']:.2f} P={100 * m['precision']:.2f} R={100 * m['recall']:.2f} "
              f"TP={m['tp']} FP={m['fp']} FN={m['fn']}", flush=True)
    print(f"eval AUROC={result['eval_auroc']} probe vs qwen dF1={result['probe_vs_qwen_dF1']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["features", "fit"], required=True)
    parser.add_argument("--variant", choices=["A", "B"], required=True)
    parser.add_argument("--eval-prompts", default="/app/local_extensions/steering/prompts.jsonl")
    parser.add_argument("--train-traces")
    parser.add_argument("--eval-traces")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--model", default="/app/models/Qwen3.8-27B-W4A16-AutoRound")
    args = parser.parse_args()
    features(args) if args.stage == "features" else fit(args)


if __name__ == "__main__":
    main()
