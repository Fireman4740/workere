"""Small optional Decisions API producer, independent of the historical LLM pipeline.

Run --smoke for synthetic API integration checks, or --inputs for a full frozen input file.
Credentials: OPENROUTER_API_KEY (or the existing OPENROUTER environment variable).
Reference: https://openrouter.ai/api/alpha/decisions
"""
from __future__ import annotations

import argparse
import json
import math
import os
import urllib.error
import urllib.request
from pathlib import Path

from benchmark.io import fingerprint, read_jsonl, write_json, write_jsonl
from benchmark.labels import CAUSED_BY, CAUSES, NOREL

MODELS = ("inception/mercury-decide:free", "respan/span-01-lite:free")
ENDPOINT = "https://openrouter.ai/api/alpha/decisions"


def request_body(model, text, source, target):
    if model not in MODELS:
        raise ValueError(f"unsupported example backend: {model}")
    state = f"Text: {text}\nEvent 1: {source}\nEvent 2: {target}"
    if model == MODELS[0]:
        questions = {"relation": {"type": "choice", "instructions":
                     "Which causal relation between the two events is supported by the text?",
                     "criteria": {"A": "event_1 causes event_2", "B": "event_2 causes event_1",
                                  "C": "neither event causes the other"}}}
    else:
        # Span accepts noul only, plain-string criteria values, and a textual state.
        questions = {key: {"type": "noul", "instructions": question,
                          "criteria": {"true": "The causal claim is supported by the text.",
                                       "false": "The causal claim is unsupported or reversed."}}
                     for key, question in (("forward", "Does event_1 cause event_2?"),
                                           ("reverse", "Does event_2 cause event_1?"))}
    return {"model": model, "state": state, "questions": questions}


def call(body):
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER")
    if not key:
        raise ValueError("set OPENROUTER_API_KEY")
    req = urllib.request.Request(ENDPOINT, data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}",
                                          "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as stream:
            return json.load(stream)
    except urllib.error.HTTPError as exc:
        # Do not dump credentials, requests or provider error bodies.
        raise RuntimeError(f"Decisions API HTTP {exc.code}") from None


def decode(model, response, threshold=0.5):
    answers = response["answers"]
    if model == MODELS[0]:
        return {"A": CAUSES, "B": CAUSED_BY, "C": NOREL}[answers["relation"]["choice"]]
    scores = [answers[key]["noul"] for key in ("forward", "reverse")]
    if any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in scores):
        raise ValueError("invalid noul probability")
    if max(scores) <= threshold or scores[0] == scores[1]:
        return NOREL
    return CAUSES if scores[0] > scores[1] else CAUSED_BY


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--inputs", help="gold-free inputs exported by benchmark inputs")
    parser.add_argument("--model", choices=MODELS, default=MODELS[0])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not math.isfinite(args.threshold) or not 0 <= args.threshold <= 1:
        parser.error("threshold must be in [0,1]")
    paths = [args.output] if args.smoke else [args.output, args.output + ".responses.jsonl",
                                             args.output + ".metadata.json"]
    if any(Path(path).exists() for path in paths):
        parser.error("output already exists; choose a fresh path before making API calls")
    if args.smoke:
        checks = []
        for model in MODELS:
            body = request_body(model, "Heavy rain caused flooding.", "rain", "flooding")
            response = call(body)
            checks.append({"requested_model": model, "request": body, "response": response,
                           "decoded_label": decode(model, response, args.threshold)})
        write_json(args.output, {"task": "synthetic-api-smoke", "is_benchmark": False,
                                 "threshold": args.threshold, "checks": checks})
        return
    rows = read_jsonl(args.inputs)
    output, raw = [], []
    for row in rows:
        triples = []
        for source, target in row["pair_list"]:
            body = request_body(args.model, row["text"], f"<{source} {row['surfaces'][source]}>",
                                f"<{target} {row['surfaces'][target]}>")
            response = call(body)
            label = decode(args.model, response, args.threshold)
            raw.append({"id": row["id"], "pair": [source, target], "response": response})
            if label != NOREL:
                triples.append([source, label, target])
        output.append({"id": row["id"], "triples": triples})
    write_jsonl(args.output, output)
    write_jsonl(args.output + ".responses.jsonl", raw)
    write_json(args.output + ".metadata.json", {"model": args.model, "threshold": args.threshold,
               "input_fingerprint": fingerprint(rows), "calls": len(raw), "endpoint": ENDPOINT})


if __name__ == "__main__":
    main()
