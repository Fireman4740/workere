"""J2: DeepSeek verifier on the uncertain band of the frozen Luna x d1 fused score.

Band membership uses only the fused score (label-free). Each document's band pairs are
chunked, and `samples` calls per chunk with independently shuffled order are averaged.
Stacked score on band pairs: (1 - w) * fused + w * judge. Primary (pre-registered): threshold
fixed at 0.39, w chosen leave-one-topic-out over `weights`.
"""
import json
import random

import numpy as np

from pipelines.judges.common import (FROZEN_THRESHOLD, Dev44, Scorer, Spend, chunks, client_from,
                                    compare, finite, last_json, run_parallel)

SYSTEM = (
    "You are an expert annotator of event causality in English news (EventStoryLine v0.9, Caselli & "
    "Vossen 2017). Event mentions are marked inline as <ID text>.\n"
    "A pair is causal when one event causes, enables, leads to, or prevents the other, in either "
    "direction, and the text supports the link directly or by clear implication. Temporal succession, "
    "co-occurrence in a passage, or vague plausibility are not causation: ask whether the effect would "
    "still have happened as described had the cause not happened.\n"
    "Judge EVENTS, not mentions: an article refers to one event several times. If the event denoted by "
    "one mention causes the event denoted by the other, the pair is causal even when the link is stated "
    "elsewhere in the article through other mentions of the same events. Two mentions of the same event "
    "are not causally related to each other.\n"
    "For each listed pair give the probability (0 to 1) that it is causal under these guidelines. "
    "Reason, then output the final answer as a JSON array with one object per listed pair: "
    '[{"pair": "A,B", "p": 0.7}, ...]. Every listed pair must appear exactly once.'
)
USER = "Document {doc_id}:\n{doc_text}\n\nPairs to judge:\n{pair_lines}"


def _requested(body):
    text = body["messages"][-1]["content"].split("Pairs to judge:\n", 1)[1]
    return {",".join(sorted(line[2:].split(" ", 1)[0].split(","))) for line in text.splitlines()
            if line.startswith("- ")}


def parse_probabilities(content, body, *, allow_missing=False):
    wanted, out = _requested(body), {}
    for row in last_json(content, list):
        if not isinstance(row, dict) or not isinstance(row.get("pair"), str) or not finite(row.get("p")):
            raise ValueError("each answer needs a string pair and a numeric p")
        if not 0 <= row["p"] <= 1:
            raise ValueError(f"probability out of range: {row['p']}")
        key = ",".join(sorted(row["pair"].replace(" ", "").split(",")))
        if key not in wanted:
            raise ValueError(f"pair not requested: {row['pair']}")
        if key in out:
            raise ValueError(f"duplicate answer for {key}")
        out[key] = float(row["p"])
    if set(out) != wanted and (not allow_missing or not out):
        raise ValueError(f"missing answers: {sorted(wanted - set(out))[:10]}")
    return out


def parse_available(content, body):
    """Retain valid partial replies for targeted completion, never for scoring."""
    return parse_probabilities(content, body, allow_missing=True)


def ask_complete(client, data, doc, pairs, sample, spend):
    """Preserve the initial request/cache; recover only omissions in separate calls.

    Completeness repair sees text + missing ids only, without scores or labels.
    Initial complete replies remain bit-for-bit cache-compatible with run one.
    """
    surface = data.suite[doc]["surfaces"]
    answers, pending = {}, list(pairs)
    usage, repairs = [], 0
    for attempt in range(4):
        lines = "\n".join(f"- {a},{b} ({surface.get(a, '')!r} / {surface.get(b, '')!r})" for a, b in pending)
        user = USER.format(doc_id=doc, doc_text=data.marked_text(doc), pair_lines=lines)
        system = SYSTEM
        if attempt:
            system += ("\nThis is a completeness repair. Return one probability for EVERY listed pair, "
                       "including pairs you judge non-causal or coreferent. Do not omit any pair.")
        reply = client.ask(system, user, sample)
        spend.add(reply)
        usage.append(reply["usage"])
        partial = parse_available(reply["content"], {"messages": [{"content": user}]})
        answers.update(partial)
        pending = [p for p in pending if ",".join(sorted(p)) not in answers]
        if not pending:
            return answers, usage, repairs
        repairs += 1
        print("J2_COMPLETENESS_REPAIR " + json.dumps({"doc": doc, "sample": sample,
              "missing_pairs": [list(p) for p in pending], "repair": repairs}), flush=True)
    raise ValueError(f"incomplete J2 judgement after targeted repairs for {doc}: {pending}")


def band_pairs(data, doc, low, high):
    return [p for p in data.unordered(doc) if low <= data.fused[doc][p if p in data.fused[doc] else p[::-1]] < high]


def run(cfg):
    data = Dev44(cfg["controls"], cfg["suites_dir"])
    low, high = cfg["band"]
    client = client_from(cfg, parse_available)
    band = {d: band_pairs(data, d, low, high) for d in data.docs}
    print("J2_BAND " + json.dumps({"unordered_pairs": sum(map(len, band.values())),
                                   "documents_with_band": sum(bool(v) for v in band.values())}), flush=True)
    spend = Spend()

    def job(doc, sample, index, pairs):
        def call():
            pairs = list(pairs_arg)
            random.Random(f"{doc}::{sample}").shuffle(pairs)
            probs, usage, repairs = ask_complete(client, data, doc, pairs, sample, spend)
            print("J2_JUDGE " + json.dumps({"doc": doc, "sample": sample, "chunk": index, "pairs": len(pairs),
                                            "completeness_repairs": repairs, "usage": usage}), flush=True)
            return doc, sample, probs
        pairs_arg = pairs
        return call

    try:
        replies = run_parallel([job(d, s, i, part) for d in data.docs
                               for i, part in enumerate(chunks(band[d], cfg["pairs_per_call"]))
                               for s in range(1, cfg["samples"] + 1)], cfg["workers"])
    finally:
        client.close()
    judge = {d: {} for d in data.docs}
    for doc, _, probs in replies:
        for key, p in probs.items():
            judge[doc].setdefault(tuple(key.split(",")), []).append(p)
    if any(len(v) != cfg["samples"] for m in judge.values() for v in m.values()):
        raise ValueError("a band pair did not receive the exact number of independent samples")
    judge = {d: {k: float(np.mean(v)) for k, v in m.items()} for d, m in judge.items()}
    return score(cfg, data, judge, spend.value)


def stacked(data, judge, w):
    def scores(doc):
        out = {}
        for (a, b), f in data.fused[doc].items():
            j = judge[doc].get(tuple(sorted((a, b))))
            out[(a, b)] = f if j is None else (1 - w) * f + w * j
        return out
    cache = {d: scores(d) for d in data.docs}
    return lambda d: cache[d]


def auroc(scores, labels):
    pos = [s for s, g in zip(scores, labels) if g]
    neg = [s for s, g in zip(scores, labels) if not g]
    if not pos or not neg:
        return None
    wins = sum((p > n) + .5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def score(cfg, data, judge, spend):
    """Annotations enter only here, after every model call has completed."""
    scorer = Scorer(data)
    base = scorer.fixed(lambda d: data.fused[d])
    weights = cfg["weights"]
    variants = {f"w={w}": stacked(data, judge, w) for w in weights}
    primary, primary_choice = scorer.lto(variants, thresholds=FROZEN_THRESHOLD)
    joint, joint_choice = scorer.lto(variants)
    band_rows = [(data.fused[d][k if k in data.fused[d] else k[::-1]], p,
                  any(data.by_doc[d].get(o, {}).get("gold", False) for o in (k, k[::-1])))
                 for d in data.docs for k, p in judge[d].items()]
    out = {"schema": "workere-judge-result/v1", "experiment": "J2 band verifier", "documents": len(data.docs),
           "band": cfg["band"], "spend": spend,
           "band_pairs": len(band_rows), "band_gold_positive": sum(g for *_, g in band_rows),
           "band_auroc": {"fused": auroc([f for f, _, _ in band_rows], [g for *_, g in band_rows]),
                          "judge": auroc([p for _, p, _ in band_rows], [g for *_, g in band_rows])},
           "frozen_fusion": compare(base, base, data.docs),
           "primary_w_lto_fixed_0.39": {**compare(primary, base, data.docs),
                                        "choices": sorted({c["variant"] for c in primary_choice.values()})},
           "secondary_w1_fixed_0.39": compare(scorer.fixed(stacked(data, judge, 1)), base, data.docs),
           "secondary_joint_lto": {**compare(joint, base, data.docs),
                                   "choices": sorted({(c["variant"], c["threshold"]) for c in joint_choice.values()})},
           "fixed_0.39_per_w": {name: compare(scorer.fixed(f), base, data.docs)["f1"] for name, f in variants.items()},
           "ap": {"frozen_fusion": scorer.ap(lambda d: data.fused[d]),
                  **{name: scorer.ap(f) for name, f in variants.items()}}}
    out["choices_by_topic"] = {"primary": primary_choice, "secondary_joint": joint_choice}
    out["per_document"] = {d: {"base": base[d], "primary": primary[d], "secondary_joint": joint[d]}
                           for d in data.docs}
    out["judge_probabilities"] = {d: {",".join(k): p for k, p in m.items()} for d, m in judge.items()}
    return out
