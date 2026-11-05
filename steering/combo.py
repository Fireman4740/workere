#!/usr/bin/env python
"""R1: post-reasoning pair features on every budgeted trace, then vote / probe / verifier
and their fusion on the 47 controlled CTB docs.

features: one prefill per (doc, trace): N13 prompt + the trace's thinking + a pair block that
          asks for yes/no after each candidate pair. At the last token of each pair line it keeps
          the layer-40 residual (probe input) and the zero-training verifier score
          log P(yes) - log P(no), read from the final layer through the model's own norm and
          lm_head (the next-token distribution at that position).
fit:      probe = logistic regression on train-doc features (doc-grouped 5-fold CV picks L2 and
          threshold); fusion = logistic regression on [probe out-of-fold logit, verifier] fitted
          on train docs; everything then applied once to the eval passes.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

import ctb
from common import read_jsonl

HEADER = ("\n\nPair check: for each candidate pair below, answer yes if the two events are causally "
          "related in the text, otherwise no.\n")
PROBE_LAYER, FINAL_LAYER = 40, 63


def pair_block(doc: dict) -> tuple[str, list[tuple[str, str]], list[int]]:
    surf = ctb.surfaces(doc)
    text, ends, pairs = HEADER, [], ctb.unordered_pairs(doc)
    for a, b in pairs:
        text += f"{a} ({surf.get(a, '?')}) / {b} ({surf.get(b, '?')}):"
        ends.append(len(text))
        text += "\n"
    return text, pairs, ends


def thinking_of(text: str) -> str | None:
    return text.split("</think>")[0] if "</think>" in text else None


def verifier_head(model, tokenizer):
    """Final norm, lm_head weight and the yes/no token ids, or None if not reachable."""
    norm = head = None
    for name, module in model.named_modules():
        if name.endswith("lm_head") and hasattr(module, "weight"):
            head = module.weight
        if name.split(".")[-1] == "norm" and "layers" not in name and hasattr(module, "weight"):
            norm = module
    if norm is None or head is None:
        return None
    ids = {}
    for word in ("yes", "no"):
        variants = {tokenizer.encode(v, add_special_tokens=False)[0] for v in (word, " " + word, word.capitalize(), " " + word.capitalize())}
        ids[word] = sorted(variants)
    return norm, head, ids


def verifier_scores(rows, head_spec) -> np.ndarray:
    import torch

    norm, head, ids = head_spec
    with torch.inference_mode():
        h = rows.to(device=head.device, dtype=head.dtype)
        normed = norm(h)
        normed = normed[0] if isinstance(normed, tuple) else normed
        logits = torch.nn.functional.linear(normed, head).float()
        yes = torch.logsumexp(logits[:, ids["yes"]], -1)
        no = torch.logsumexp(logits[:, ids["no"]], -1)
        return (yes - no).cpu().numpy()


def features(args) -> None:
    from interventions import capture_rows, find_layer
    from offline import build_llm, render

    suite, prompt = ctb.load_suite(), ctb.load_prompt()
    evals = set(ctb.eval_ids())
    eval_messages = {r["doc_id"]: r["messages"] for r in read_jsonl(args.eval_prompts) if int(r["pass"]) == 0}
    jobs = []  # (split, trace name, doc_id, thinking)
    for row in read_jsonl(args.train_traces):
        if (t := thinking_of(row["text"])) is not None and row["doc_id"] not in evals:
            jobs.append(("train", "T0", row["doc_id"], t))
    for row in read_jsonl(args.gen):
        if (t := thinking_of(row["text"])) is not None and row["doc_id"] in evals:
            jobs.append(("eval", row["condition"], row["doc_id"], t))
    llm, model, tokenizer = build_llm(args.model, args.max_model_len, args.gpu_memory_utilization, eager=True)
    taps = {"probe": (find_layer(model, PROBE_LAYER), "resid"), "final": (find_layer(model, FINAL_LAYER), "resid")}
    head_spec = verifier_head(model, tokenizer)
    print(f"verifier head {'found' if head_spec else 'NOT found'}; {len(jobs)} traces", flush=True)
    feats, verif, meta = [], [], []
    for number, (split, trace, doc_id, thinking) in enumerate(jobs, 1):
        doc = suite[doc_id]
        messages = eval_messages[doc_id] if doc_id in evals else ctb.messages_for(doc, prompt)
        head = render(tokenizer, messages) + thinking
        block, pairs, ends = pair_block(doc)
        full = head + block
        offsets = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        positions, index = [], 0
        for end in (len(head) + e for e in ends):
            while index < len(offsets) and offsets[index][1] < end:
                index += 1
            positions.append(min(index, len(offsets) - 1))
        if len(offsets) >= args.max_model_len - 2:
            print(f"skip {trace} {doc_id}: {len(offsets)} tokens", flush=True)
            continue
        store = capture_rows(llm, full, taps, positions)
        feats.append(store["probe"].numpy().astype(np.float16))
        verif.append(verifier_scores(store["final"], head_spec) if head_spec else np.full(len(pairs), np.nan))
        gold = ctb.causal_pairs(doc)
        meta += [{"split": split, "trace": trace, "doc": doc_id, "a": a, "b": b, "gold": (a, b) in gold} for a, b in pairs]
        if number % 20 == 0:
            print(f"[{number}/{len(jobs)}] {split} {trace} {doc_id}", flush=True)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "features.npz", probe=np.concatenate(feats), verifier=np.concatenate(verif))
    (out / "meta.json").write_text(json.dumps(meta))


def fit(args) -> None:
    from compare import paired_delta
    from probe import auroc, best_threshold, standardise, train_logreg

    out = Path(args.out_dir)
    data = np.load(out / "features.npz")
    meta = json.loads((out / "meta.json").read_text())
    x, ver = data["probe"].astype(np.float32), data["verifier"].astype(np.float64)
    y = np.array([m["gold"] for m in meta])
    split = np.array([m["split"] for m in meta])
    docs = np.array([m["doc"] for m in meta])
    traces = np.array([m["trace"] for m in meta])
    train = split == "train"
    train_docs = sorted(set(docs[train]))
    folds = {d: i % 5 for i, d in enumerate(random.Random("r1-folds").sample(train_docs, len(train_docs)))}
    fold = np.array([folds.get(d, -1) for d in docs])
    best = None
    for l2 in (1e-3, 1e-2, 1e-1):
        oof = np.zeros(len(y))
        for k in range(5):
            tr, te = train & (fold != k), train & (fold == k)
            a, b_ = standardise(x[tr], x[te])
            w, b = train_logreg(a, y[tr], l2)
            oof[te] = b_ @ w + b
        f1, thr = best_threshold(oof[train], y[train])
        print(f"probe cv l2={l2} F1={100 * f1:.2f} AUROC={auroc(oof[train], y[train]):.4f}", flush=True)
        if best is None or f1 > best[0]:
            best = (f1, thr, l2, oof.copy())
    _, thr, l2, oof = best
    xtr, xall = standardise(x[train], x)
    w, b = train_logreg(xtr, y[train], l2)
    probe = xall @ w + b
    probe[train] = oof[train]
    finite = np.isfinite(ver)
    has_ver = finite[train].all() and finite.any()
    ver_thr = best_threshold(ver[train], y[train])[1] if has_ver else None
    if has_ver:
        z = np.stack([probe, ver], 1)
        zm, zs = z[train].mean(0), z[train].std(0) + 1e-6
        fw, fb = train_logreg((z[train] - zm) / zs, y[train], 1e-3)
        fused = ((z - zm) / zs) @ fw + fb
        fthr = best_threshold(fused[train], y[train])[1]
    reference = json.loads((ctb.DATA / "reference.json").read_text())
    ids = reference["qwen_ids"]
    gold_ord = {d: {tuple(k.split(",", 1)) for k, v in reference["gold"][d].items() if v != "norel"} for d in ids}
    qwen7 = {d: {tuple(k.split(",", 1)) for k, v in reference["qwen"][d].items() if v != "norel"} for d in ids}

    def counts(pred_unordered: dict) -> dict:
        res = {}
        for d in ids:
            pred = {p for a, b in pred_unordered.get(d, set()) for p in ((a, b), (b, a))}
            res[d] = {"tp": len(pred & gold_ord[d]), "fp": len(pred - gold_ord[d]), "fn": len(gold_ord[d] - pred)}
        return res

    def f1(c):
        tp = sum(v["tp"] for v in c.values()); fp = sum(v["fp"] for v in c.values()); fn = sum(v["fn"] for v in c.values())
        return round(100 * 2 * tp / (2 * tp + fp + fn), 2), tp, fp, fn

    from eci.parse import ParseError, parse_triples
    answers = {}
    for row in read_jsonl(args.gen):
        pred = None
        if row.get("finish_reason") != "length" and "</think>" in row["text"]:
            try:
                pred = {tuple(sorted((a, b))) for a, lab, b in parse_triples(row["text"]) if lab != "norel"}
            except (ParseError, ValueError):
                pred = None
        answers.setdefault(row["condition"], {})[row["doc_id"]] = pred
    conds = sorted(answers)
    ev = (split == "eval") & np.isin(docs, ids)
    systems = {"qwen 7-pass API": {d: {tuple(sorted(p)) for p in qwen7[d]} for d in ids}}

    def vote(cs, thr_=0.5):
        out_ = {}
        for d in ids:
            got = [answers[c].get(d) for c in cs if answers[c].get(d) is not None]
            cnt = {}
            for g in got:
                for p in g:
                    cnt[p] = cnt.get(p, 0) + 1
            out_[d] = {p for p, n in cnt.items() if got and n / len(got) >= thr_ - 1e-9}
        return out_

    for k in range(1, len(conds) + 1):
        systems[f"vote@{k}"] = vote(conds[:k])

    def per_pair(score, mask, thr_):
        out_ = {}
        for m, s, keep in zip(meta, score, mask):
            if keep and s >= thr_:
                out_.setdefault(m["doc"], set()).add((m["a"], m["b"]))
        return out_

    first = conds[0]
    systems["probe@1"] = per_pair(probe, ev & (traces == first), thr)
    mean_scores = {}
    for m, s, keep in zip(meta, probe, ev):
        if keep:
            mean_scores.setdefault((m["doc"], m["a"], m["b"]), []).append(s)
    systems[f"probe-mean@{len(conds)}"] = {}
    for (d, a, b), ss in mean_scores.items():
        if np.mean(ss) >= thr:
            systems[f"probe-mean@{len(conds)}"].setdefault(d, set()).add((a, b))
    if has_ver:
        systems["verifier@1 zero-shot (P(yes)>P(no))"] = per_pair(ver, ev & (traces == first), 0.0)
        systems["verifier@1 (train-calibrated threshold)"] = per_pair(ver, ev & (traces == first), ver_thr)
        systems["fusion@1 (probe+verifier, train-fitted)"] = per_pair(fused, ev & (traces == first), fthr)
        fmean = {}
        for m, s, keep in zip(meta, fused, ev):
            if keep:
                fmean.setdefault((m["doc"], m["a"], m["b"]), []).append(s)
        systems[f"fusion-mean@{len(conds)}"] = {}
        for (d, a, b), ss in fmean.items():
            if np.mean(ss) >= fthr:
                systems[f"fusion-mean@{len(conds)}"].setdefault(d, set()).add((a, b))
    vk = systems[f"vote@{len(conds)}"]
    pm = systems[f"probe-mean@{len(conds)}"]
    systems["vote OR probe-mean"] = {d: vk.get(d, set()) | pm.get(d, set()) for d in ids}
    systems["vote AND probe-mean"] = {d: vk.get(d, set()) & pm.get(d, set()) for d in ids}
    base = counts(systems["qwen 7-pass API"])
    report = {"probe": {"l2": l2, "threshold": thr}, "verifier_threshold": ver_thr,
              "eval_auroc": {"probe": auroc(probe[ev], y[ev])} | ({"verifier": auroc(ver[ev], y[ev])} if has_ver else {}),
              "unanswered": {c: sum(v is None for d, v in answers[c].items() if d in ids) for c in conds}, "systems": {}}
    for name, pred in systems.items():
        c = counts(pred)
        d = paired_delta(c, base, 10000) if name != "qwen 7-pass API" else (0.0, 0.0, 0.0)
        report["systems"][name] = {"f1": f1(c), "vs_qwen7": [round(v, 2) for v in d]}
        print(f"{name:42s} F1={f1(c)[0]:6.2f} TP={f1(c)[1]} FP={f1(c)[2]} FN={f1(c)[3]}  vs qwen7 {d[0]:+.2f} [{d[1]:+.2f}, {d[2]:+.2f}]", flush=True)
    print("eval AUROC", report["eval_auroc"], "unanswered", report["unanswered"], flush=True)
    (out / "result.json").write_text(json.dumps(report, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["features", "fit"], required=True)
    parser.add_argument("--gen", required=True, help="budgeted multi-pass generations on eval docs")
    parser.add_argument("--train-traces")
    parser.add_argument("--eval-prompts", default="/app/local_extensions/steering/prompts.jsonl")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-model-len", type=int, default=24576)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--model", default="/app/models/Qwen3.8-27B-W4A16-AutoRound")
    args = parser.parse_args()
    features(args) if args.stage == "features" else fit(args)


if __name__ == "__main__":
    main()
