"""Dense pair decisions with frozen Qwen; no internal-head lookup or gradients.

V1: direct binary question. V2: same question + retrieved 2 positive/2 negative
examples from the historical 20-document demo bank. All remaining 113 non-N13
documents calibrate thresholds; N13 labels never enter prompts or thresholds.
Both variants score every candidate, including candidates absent from all votes.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import time

from ctb import load_suite, load_prompt, unordered_pairs, causal_pairs, eval_ids

HERE = Path(__file__).resolve().parent
MODEL = "/app/models/Qwen3.8-27B-W4A16-AutoRound"
GROUPS = [
    {"because", "due", "since", "therefore", "thus", "result"},
    {"to", "purpose", "order", "attempt", "intend", "intended"},
    {"allow", "allowed", "allows", "enable", "enabled", "help", "helped"},
    {"prevent", "prevented", "block", "blocked", "stop", "stopped"},
    {"cause", "caused", "causing", "force", "forced", "lead", "led"},
]


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2))


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def excerpt(doc, pair):
    """Exact annotated sentence plus one neighbouring sentence on either side."""
    sent = int(doc["sentence_of"][pair[0]])
    assert sent == int(doc["sentence_of"][pair[1]])
    left = doc["sentences"][max(0, sent - 1)][0]
    right = doc["sentences"][min(len(doc["sentences"]) - 1, sent + 1)][1]
    marks = {doc["first_token"][p]: (p, doc["surfaces"][p]) for p in pair}
    return " ".join(f"<{marks[i][0]} {token}>" if i in marks else token
                    for i, token in enumerate(doc["tokens"]) if left <= i < right)


def features(doc, pair):
    text = excerpt(doc, pair)
    words = set(re.findall(r"[a-z]+", text.lower()))
    anchors = {i for i, group in enumerate(GROUPS) if words & group}
    distance = abs(doc["first_token"][pair[0]] - doc["first_token"][pair[1]])
    return words, anchors, distance


def retrieve(doc, pair, bank):
    words, anchors, distance = features(doc, pair)
    ranked = []
    for b in bank:
        score = (len(words & b["words"]) / max(1, len(words | b["words"]))
                 + 0.4 * len(anchors & b["anchors"])
                 - 0.02 * abs(math.log1p(distance) - math.log1p(b["distance"])))
        ranked.append((-score, b["doc_id"], b["pair"], b))
    ranked.sort(key=lambda x: x[:3])
    picked = []
    for label in (True, False):
        used_docs = set()
        for *_, b in ranked:
            if b["label"] == label and b["doc_id"] not in used_docs:
                picked.append(b)
                used_docs.add(b["doc_id"])
                if len(used_docs) == 2:
                    break
    assert len(picked) == 4
    # Interleave positive and negative examples, identical across label permutations.
    return [picked[i] for i in (0, 2, 1, 3)]


def make_messages(rules, context, pair, surfaces, demos, reverse=False, null=False):
    yes, no = ("B", "A") if reverse else ("A", "B")
    system = (rules + "\n\nClassify only the requested pair: is there an annotated causal "
              "relation in either direction? Direction is not being scored here. "
              f"Answer {yes} for causal, {no} for no causal relation. "
              "Return exactly one letter, with no explanation or JSON.")
    messages = [{"role": "system", "content": system}]
    def question(text, ids, names):
        return (f"Text:\n{text}\n\nPair: {ids[0]} ({names[0]}) and {ids[1]} ({names[1]}). "
                "Is there a causal relation between these two events?")
    for d in demos:
        messages.extend([{"role": "user", "content": question(d["text"], d["pair"], d["surfaces"])},
                         {"role": "assistant", "content": yes if d["label"] else no}])
    messages.append({"role": "user", "content": question(
        "N/A" if null else context, ("event1", "event2") if null else pair,
        ("N/A", "N/A") if null else surfaces)})
    return messages


def prepare(out, variant, resume_from=None):
    suite, prompt = load_suite(), load_prompt()
    excluded = set(eval_ids())
    bank_ids = {d["id"] for d in prompt["demos"]}
    cal_ids = sorted(set(suite) - excluded - bank_ids)
    ref = json.loads((HERE / "data/reference.json").read_text())
    evaluation = sorted(ref["qwen_ids"])
    assert not bank_ids & excluded and len(evaluation) == 47
    rules = prompt["system"].split("# Output Format")[0]
    bank = []
    for doc_id in sorted(bank_ids):
        doc = suite[doc_id]
        for pair in unordered_pairs(doc):
            words, anchors, distance = features(doc, pair)
            bank.append(dict(doc_id=doc_id, pair=pair, label=pair in causal_pairs(doc),
                             text=excerpt(doc, pair), surfaces=[doc["surfaces"][p] for p in pair],
                             words=words, anchors=anchors, distance=distance))
    rows, requests, request_seen = [], [], set()
    for split, doc_ids in (("calibration", cal_ids), ("evaluation", evaluation)):
        for doc_id in doc_ids:
            doc = suite[doc_id]
            for pair in unordered_pairs(doc):
                demos = retrieve(doc, pair, bank) if variant == "V2" else []
                keys = []
                for reverse in (False, True):
                    for null in (False, True):
                        messages = make_messages(rules, excerpt(doc, pair), pair,
                                                 [doc["surfaces"][p] for p in pair], demos, reverse, null)
                        key = hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()
                        if key not in request_seen:
                            requests.append(dict(key=key, messages=messages, reverse=reverse, null=null))
                            request_seen.add(key)
                        keys.append(key)
                # Labels are kept in this manifest only; requests contain only bank labels.
                rows.append(dict(doc_id=doc_id, pair=pair, split=split, label=pair in causal_pairs(doc),
                                 keys=keys, demo_ids=[d["doc_id"] for d in demos]))
    manifest = dict(variant=variant, checkpoint=MODEL, bank_ids=sorted(bank_ids),
                    calibration_ids=cal_ids, evaluation_ids=evaluation, excluded_ids=sorted(excluded),
                    rows=rows, commit=__import__("os").environ.get("ORX_COMMIT"))
    dump(out / "manifest.json", manifest)
    (out / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in requests))
    if resume_from is not None:
        assert (resume_from / "requests.jsonl").read_bytes() == (out / "requests.jsonl").read_bytes(), "resume request mismatch"
        source_manifest = json.loads((resume_from / "manifest.json").read_text())
        assert {k:v for k,v in source_manifest.items() if k != "commit"} == {k:v for k,v in manifest.items() if k != "commit"}, "resume manifest mismatch"
        for name in ("scores.jsonl", "smoke.json"):
            if (resume_from / name).exists():
                shutil.copy2(resume_from / name, out / name)
        dump(out / "resume.json", dict(source=str(resume_from), request_identity_verified=True))
        print("RESUME verified identical requests and split; copied existing scores", flush=True)
    print("PREPARED", json.dumps(dict(variant=variant, bank_documents=len(bank_ids),
          calibration_documents=len(cal_ids), evaluation_documents=len(evaluation),
          pairs=len(rows), requests=len(requests), splits=dict(Counter(r["split"] for r in rows)))), flush=True)


def infer(out):
    import os
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")
    from vllm import LLM, SamplingParams
    from offline import ENGINE_EXTRA
    requests = jsonl(out / "requests.jsonl")
    print("ENGINE_CONFIG", json.dumps(dict(model=MODEL, model_config_sha256=
        hashlib.sha256((Path(MODEL)/"config.json").read_bytes()).hexdigest(),
        max_model_len=8192, max_num_seqs=16, max_num_batched_tokens=2048,
        thinking=False, generated_tokens_per_query=1, gpu_memory_utilization=.90)), flush=True)
    llm = LLM(model=MODEL, tokenizer=MODEL, trust_remote_code=True,
              max_model_len=8192, max_num_seqs=16, max_num_batched_tokens=2048,
              enable_chunked_prefill=True, enable_prefix_caching=False,
              gpu_memory_utilization=0.90, enforce_eager=True, disable_log_stats=True,
              **ENGINE_EXTRA)
    tok = llm.get_tokenizer()
    token_ids = [tok.encode(letter, add_special_tokens=False) for letter in ("A", "B")]
    assert all(len(ids) == 1 for ids in token_ids), token_ids
    label_ids = [ids[0] for ids in token_ids]
    params = SamplingParams(temperature=0, max_tokens=1, logprobs=2,
                            logprob_token_ids=label_ids, seed=20261106)
    print("NATIVE_LABEL_IDS", dict(zip(("A", "B"), label_ids)), flush=True)
    def run(batch):
        texts = [tok.apply_chat_template(r["messages"], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False) for r in batch]
        # No silent truncation of the prompt or exclusion from the scored set.
        sizes = [len(tok.encode(t, add_special_tokens=False)) for t in texts]
        assert max(sizes) < 8191, max(sizes)
        if any("<think>" in t and "</think>" not in t.rsplit("<think>", 1)[1] for t in texts):
            raise RuntimeError("chat template did not close thinking for native-label classification")
        results = llm.generate(texts, params, use_tqdm=False)
        got = []
        for req, result, size in zip(batch, results, sizes):
            probs = result.outputs[0].logprobs[0]
            if any(i not in probs for i in label_ids):
                raise RuntimeError("native requested-label logprobs missing; do not silently skip verifier")
            a, b = (float(probs[i].logprob) for i in label_ids)
            assert math.isfinite(a) and math.isfinite(b), (a, b)
            got.append(dict(key=req["key"], logp_a=a, logp_b=b,
                            margin=(b-a if req["reverse"] else a-b),
                            prompt_tokens=size, generated=result.outputs[0].text))
        return got
    smoke = requests[:8]
    first, second = run(smoke), run(smoke)
    drift = max(abs(a["margin"] - b["margin"]) for a, b in zip(first, second))
    dump(out / "smoke.json", dict(max_margin_drift=drift, first=first, repeat=second,
                                  label_ids=label_ids, vllm=__import__("vllm").__version__))
    assert drift < 0.1, f"A/A margin drift {drift}"
    print(f"SMOKE passed native labels, template, finite scores; A/A max margin drift={drift:.6f}", flush=True)
    path = out / "scores.jsonl"
    completed = {r["key"] for r in jsonl(path)} if path.exists() else set()
    todo = [r for r in requests if r["key"] not in completed]
    start = time.monotonic()
    with path.open("a") as handle:
        for i in range(0, len(todo), 64):
            for r in run(todo[i:i+64]):
                handle.write(json.dumps(r) + "\n")
            handle.flush()
            print(f"SCORED {len(completed)+min(i+64,len(todo))}/{len(requests)} "
                  f"elapsed={time.monotonic()-start:.1f}s", flush=True)


def best_threshold(rows, scores):
    # Exact F1 sweep on calibration only. Ties favour the higher threshold.
    groups = defaultdict(list)
    for r, s in zip(rows, scores):
        groups[s].append(bool(r["label"]))
    total = sum(r["label"] for r in rows)
    tp = fp = 0
    best = (0.0, max(groups) + 1.0)
    for threshold in sorted(groups, reverse=True):
        tp += sum(groups[threshold])
        fp += len(groups[threshold]) - sum(groups[threshold])
        f1 = 2*tp / max(1, tp+fp+total)
        if f1 > best[0]:
            best = (f1, threshold)
    return best[1], best[0]


def cached_votes(paths, suite, ids):
    from eci.parse import parse_triples
    readings = defaultdict(dict)
    for path in paths:
        assert path.exists(), f"required cached readings unavailable: {path}"
        for r in jsonl(path):
            d = r["doc_id"]
            if d not in ids:
                continue
            key = (str(path), r["condition"], int(r["pass"]))
            assert key not in readings[d], f"duplicate completed reading {d} {key}"
            text = r.get("text", "")
            pred = None
            if r.get("finish_reason") in (None, "stop") and "</think>" in text:
                try:
                    triples = parse_triples(text.rsplit("</think>", 1)[1])
                    pred = {tuple(sorted((a,b))) for a, lab, b in triples if lab != "norel"}
                    pred &= set(unordered_pairs(suite[d]))
                except ValueError:
                    pass
            readings[d][key] = pred
    result = {}
    for d in ids:
        assert len(readings[d]) == 5 * len(paths), (d, len(readings[d]), paths)
        valid = [r for r in readings[d].values() if r is not None]
        votes = Counter(p for r in valid for p in r)
        result[d] = {p: votes[p]/len(valid) if valid else 0 for p in unordered_pairs(suite[d])}
    return result


def ranking_metrics(labels, scores):
    """Exact binary AUROC (half credit for ties) and non-interpolated AP."""
    groups = defaultdict(lambda: [0, 0])
    for y, s in zip(labels, scores):
        groups[s][int(bool(y))] += 1
    positives = sum(labels)
    negatives = len(labels) - positives
    assert positives and negatives, "ranking metrics require both classes"
    below = auc = 0.0
    for s in sorted(groups):
        n, p = groups[s]
        auc += p * (below + n/2)
        below += n
    tp = fp = ap = 0.0
    for s in sorted(groups, reverse=True):
        n, p = groups[s]
        tp += p
        fp += n
        ap += (p/positives) * tp/(tp+fp)
    return auc/(positives*negatives), ap


def complete(out):
    if not (out / "scores.jsonl").exists():
        return False
    rows = jsonl(out / "scores.jsonl")
    keys = {r["key"] for r in rows}
    assert len(keys) == len(rows), "duplicate cached scores"
    return keys == {r["key"] for r in jsonl(out / "requests.jsonl")}


def score(out, work):
    from compare import paired_delta
    m = json.loads((out / "manifest.json").read_text())
    raw = {r["key"]: r for r in jsonl(out / "scores.jsonl")}
    rows = m["rows"]
    assert all(k in raw for r in rows for k in r["keys"]), "incomplete pair coverage"
    suite = load_suite()
    ids = m["evaluation_ids"]
    gold = {d: causal_pairs(suite[d]) for d in ids}
    ref = json.loads((HERE / "data/reference.json").read_text())
    qwen = {d: {tuple(sorted(k.split(","))) for k,v in ref["qwen"][d].items() if v != "norel"} for d in ids}
    n1 = cached_votes([work / "out/N1/gen.jsonl"], suite, ids)
    pool = cached_votes([work / "out/N1/gen.jsonl", work / "out/P3/gen.jsonl"], suite, ids)
    def counts(pred):
        # Doubling unordered counts exactly matches the historical binary ordered-pair metric.
        return {d: dict(tp=2*len(pred[d]&gold[d]), fp=2*len(pred[d]-gold[d]),
                        fn=2*len(gold[d]-pred[d])) for d in ids}
    def metrics(c):
        tp, fp, fn = [sum(r[k] for r in c.values()) for k in ("tp", "fp", "fn")]
        return dict(f1=200*tp/max(1,2*tp+fp+fn), precision=100*tp/max(1,tp+fp),
                    recall=100*tp/max(1,tp+fn), tp=tp, fp=fp, fn=fn)
    baseline = counts(qwen)
    pooled = counts({d: {p for p,v in pool[d].items() if v >= .5} for d in ids})
    report = dict(variant=m["variant"], calibration_documents=len(m["calibration_ids"]),
                  evaluation_documents=len(ids), baseline=metrics(baseline),
                  cached_n1_p3=metrics(pooled), systems={})
    all_preds = {}
    eval_rows = [r for r in rows if r["split"] == "evaluation"]
    cal_rows = [r for r in rows if r["split"] == "calibration"]
    def values(r):
        a,b,c,d = [raw[k]["margin"] for k in r["keys"]]
        return {"raw": (a+c)/2, "centered": (a-b+c-d)/2,
                "permutation_gap": abs(a-c)}
    primary = "raw"  # frozen primary, centered is a secondary calibration diagnostic
    for method in ("raw", "centered"):
        cal_scores = [values(r)[method] for r in cal_rows]
        eval_scores = [values(r)[method] for r in eval_rows]
        threshold, cal_f1 = best_threshold(cal_rows, cal_scores)
        pair_pred = {d: set() for d in ids}
        for r,s in zip(eval_rows, eval_scores):
            if s >= threshold:
                pair_pred[r["doc_id"]].add(tuple(r["pair"]))
        # Fixed routing; no oracle/test threshold optimisation or learned fusion.
        hybrids = {}
        for name, votes in (("n1", n1), ("n1_p3", pool)):
            hybrids[name] = {d: {p for p,v in votes[d].items()
                                 if v >= .8 or p in pair_pred[d]} for d in ids}
        for name, pred in [(method, pair_pred)] + [(method+"_hybrid_"+n,p) for n,p in hybrids.items()]:
            c = counts(pred)
            recovery = sum(p in gold[d] and pool[d][p] == 0 for d in ids for p in pred[d])
            zero_fp = sum(p not in gold[d] and pool[d][p] == 0 for d in ids for p in pred[d])
            report["systems"][name] = dict(**metrics(c), threshold=threshold,
                calibration_f1=100*cal_f1, vs_qwen7=paired_delta(c,baseline,10000,seed=42),
                vs_cached_n1_p3=paired_delta(c,pooled,10000,seed=42),
                recovered_zero_vote_links=recovery, zero_vote_false_links=zero_fp,
                document_counts=c)
            all_preds[name] = {d: sorted(pred[d]) for d in ids}
            print("RESULT", name, json.dumps({k:v for k,v in report["systems"][name].items()
                                              if k != "document_counts"}), flush=True)
        auc, ap = ranking_metrics([r["label"] for r in eval_rows], eval_scores)
        report["systems"][method].update(auroc=auc, average_precision=ap)
    report["primary"] = primary
    report["mean_label_permutation_margin_gap"] = sum(values(r)["permutation_gap"] for r in eval_rows)/len(eval_rows)
    report["scope"] = "CTB47 reused development set; binary presence, not direction; all candidates scored"
    report["hashes"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in (out/"manifest.json",out/"requests.jsonl",out/"scores.jsonl")}
    dump(out / "report.json", report)
    dump(out / "predictions.json", all_preds)
    print("FINAL", json.dumps({k:v for k,v in report.items() if k != "systems"}), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("prepare", "infer", "score", "complete"))
    p.add_argument("--variant", choices=("V1", "V2"), required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--work", type=Path, default=Path("/app/local_extensions/steering/mech"))
    p.add_argument("--resume-from", type=Path)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage == "prepare": prepare(args.out, args.variant, args.resume_from)
    elif args.stage == "infer": infer(args.out)
    elif args.stage == "complete": raise SystemExit(0 if complete(args.out) else 1)
    else: score(args.out, args.work)


if __name__ == "__main__":
    main()
