"""Zero-shot "decision model" baselines (Laya, TypeSafe Jev) on intra-sentence ECI.

Every unordered pair of event mentions that share a sentence is one classification: the state is
the sentence with both mentions wrapped in « », plus the two surfaces; the question is one of the
VARIANTS below. The models return a probability per option, so each pair gets a causal score
(1 - P(no relation)) and, for `dir3`, a direction.

    predict   write one jsonl row per pair (resumable: rows already in --out are skipped)
    score     metrics tables from the jsonl files

Laya runs locally (`import laya`; torch + transformers). Jev is TypeSafe's hosted API: set
JEV_API_KEY. The Jev backend follows the documented POST /v1/systemone shape but has not been
run yet — start it with --limit 200 to measure credits per pair before a full run.

Data: the published `Nofing/*-standard-eci-workere` repos (event_1 = the mention that comes
first in the text). ESC's 23 direction-less `unlabelled_pairs` are left out of the evaluation.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

REPOS = {
    "esc": "Nofing/EventStoryLine-0.9-standard-eci-workere",
    "ctb": "Nofing/CausalTimeBank-standard-eci-workere",
    "maven": "Nofing/MAVEN-ERE-standard-eci-workere",
}

INSTR = "The text marks two events with « ». Is one of them a cause of the other?"
NLI_INSTR = "Is the claim supported by the text?"


def as_json(state: dict):
    return state


def as_claim(state: dict) -> str:
    """NLI framing: the pair becomes a claim appended to the sentence."""
    return (f"{state['text']}\n\nClaim: one of the two marked events, \"{state['event_1']}\" or "
            f"\"{state['event_2']}\", is a cause of the other.")


# key -> (question, the option meaning "no causal relation", option meaning event_1 -> event_2,
#         option meaning event_2 -> event_1, state renderer). Option keys are neutral letters, as
# the Laya card recommends over `noul` (whose true/false labels can dominate the answer).
VARIANTS = {
    "dir3": ({"type": "choice", "instructions": INSTR,
              "criteria": {"A": "yes: event_1 causes event_2",
                           "B": "yes: event_2 causes event_1",
                           "C": "no: neither event causes the other"}}, "C", "A", "B", as_json),
    "bin2": ({"type": "choice", "instructions": INSTR,
              "criteria": {"A": "yes: one of the two events causes the other",
                           "B": "no: the two events are not causally related"}}, "B", None, None, as_json),
    # bin2 with the options swapped: controls for a preference for the first option.
    "bin2swap": ({"type": "choice", "instructions": INSTR,
                  "criteria": {"A": "no: the two events are not causally related",
                               "B": "yes: one of the two events causes the other"}}, "A", None, None, as_json),
    "nli2": ({"type": "choice", "instructions": NLI_INSTR,
              "criteria": {"A": "yes, the text supports the claim",
                           "B": "no, the text does not support the claim"}}, "B", None, None, as_claim),
    "nli2swap": ({"type": "choice", "instructions": NLI_INSTR,
                  "criteria": {"A": "no, the text does not support the claim",
                               "B": "yes, the text supports the claim"}}, "A", None, None, as_claim),
}


# ── Data ─────────────────────────────────────────────────────────────────────

def sentence_state(tokens, sent, span1, span2) -> dict:
    start, end = sent
    out = []
    for t in range(start, end):
        if t in (span1[0], span2[0]):
            out.append("«")
        out.append(tokens[t])
        if t in (span1[-1], span2[-1]):
            out.append("»")
    s1 = " ".join(tokens[t] for t in span1)
    s2 = " ".join(tokens[t] for t in span2)
    return {"text": " ".join(out), "event_1": s1, "event_2": s2}


def intra_pairs(dataset: str, split: str, max_docs: int, doc_seed: int):
    """Yield one dict per intra-sentence unordered mention pair."""
    from datasets import load_dataset
    ds = load_dataset(REPOS[dataset], split=split)
    order = list(range(len(ds)))
    if max_docs > 0:
        random.Random(doc_seed).shuffle(order)
        order = sorted(order[:max_docs])
    for d in order:
        row = ds[d]
        tokens, spans, sents = row["tokens"], row["spans"], row["sentences"]
        sent_of = []
        for sp in spans:
            sent_of.append(next(k for k, (a, b) in enumerate(sents) if a <= sp[0] < b))
        causes = {(int(c), int(e)) for c, e in row["relations"]["causes"]}
        skip = {frozenset(map(int, p)) for p in (row.get("unlabelled_pairs") or [])}
        idx = sorted(range(len(spans)), key=lambda m: (spans[m][0], spans[m][-1]))
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                i, j = idx[a], idx[b]
                if sent_of[i] != sent_of[j] or frozenset((i, j)) in skip:
                    continue
                gold_dir = "12" if (i, j) in causes else "21" if (j, i) in causes else None
                yield {"doc": row["id"], "i": i, "j": j, "sent": sent_of[i],
                       "gold": int(gold_dir is not None), "gold_dir": gold_dir,
                       "state": sentence_state(tokens, sents[sent_of[i]], spans[i], spans[j])}


# ── Backends ─────────────────────────────────────────────────────────────────

class LayaBackend:
    def __init__(self, checkpoint: str, threads: int, bf16: bool, forward_batch: int):
        os.environ.setdefault("USE_TF", "0")
        self.forward_batch = forward_batch
        import torch
        import laya
        torch.set_num_threads(threads)
        sub = None if checkpoint == "english" else checkpoint
        self.agent = laya.load("convaiinnovations/laya", subfolder=sub, device="cpu")
        if bf16:  # CPU autocast; laya leaves AMP off on CPU. Probabilities move by <0.005.
            self.agent.dtype, self.agent.amp_enabled = torch.bfloat16, True
        self.name = f"laya-{checkpoint}"

    def predict(self, states, question):
        if 0 < self.forward_batch < len(states):  # length-sorted forward passes, less padding
            out = self.agent.predict_batch(states, {"q": question}, batch_size=self.forward_batch,
                                           sort_by_length=True)
        else:
            out = self.agent.predict_batch(states, {"q": question}, batch_size=len(states))
        return [r["answers"]["q"]["probabilities"] for r in out]


class JevBackend:
    """TypeSafe Jev over HTTP. Untested: no API key was available when this was written."""
    URL = "https://thejevai.com/v1/systemone"

    def __init__(self, model: str, concurrency: int):
        import requests  # noqa: F401  (fail early if missing)
        self.key = os.environ["JEV_API_KEY"]
        self.model, self.concurrency, self.name = model, concurrency, model.replace("/", "_")

    def _one(self, state, question):
        import requests
        for attempt in range(5):
            r = requests.post(self.URL, timeout=60,
                              headers={"Authorization": f"Bearer {self.key}"},
                              json={"model": self.model, "state": state, "questions": {"q": question}})
            if r.status_code == 200:
                return r.json()["answers"]["q"]["probabilities"]
            if r.status_code not in (429, 500, 502, 503, 504):
                raise RuntimeError(f"Jev {r.status_code}: {r.text[:300]}")
            time.sleep(2 ** attempt)
        raise RuntimeError("Jev: retries exhausted")

    def predict(self, states, question):
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(self.concurrency) as ex:
            return list(ex.map(lambda s: self._one(s, question), states))


# ── predict ──────────────────────────────────────────────────────────────────

def cmd_predict(args):
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        with out.open() as fh:
            done = {(r["doc"], r["i"], r["j"]) for r in map(json.loads, fh)}
    pairs = [p for p in intra_pairs(args.dataset, args.split, args.max_docs, args.doc_seed)
             if not args.gold_only or p["gold"]]
    if args.limit > 0:
        pairs = pairs[:args.limit]
    todo = [p for p in pairs if (p["doc"], p["i"], p["j"]) not in done]
    print(f"{args.dataset}/{args.split}: {len(pairs)} pairs, {len(todo)} to do -> {out}", flush=True)
    if not todo:
        return
    backend = (LayaBackend(args.checkpoint, args.threads, args.bf16, args.forward_batch)
               if args.backend == "laya"
               else JevBackend(args.jev_model, args.concurrency))
    question, render = VARIANTS[args.variant][0], VARIANTS[args.variant][4]
    t0 = time.time()
    with out.open("a") as fh:
        for k in range(0, len(todo), args.batch_size):
            chunk = todo[k:k + args.batch_size]
            probs = backend.predict([render(p["state"]) for p in chunk], question)
            for p, pr in zip(chunk, probs):
                fh.write(json.dumps({**{x: p[x] for x in ("doc", "i", "j", "sent", "gold", "gold_dir")},
                                     "probs": pr, "model": backend.name, "variant": args.variant}) + "\n")
            fh.flush()
            n = k + len(chunk)
            if n % (args.batch_size * 20) < args.batch_size or n == len(todo):
                el = time.time() - t0
                print(f"  {n}/{len(todo)}  {n / el:.1f} pairs/s  eta {(len(todo) - n) / (n / el) / 60:.0f} min",
                      flush=True)


# ── score ────────────────────────────────────────────────────────────────────

def load_rows(path):
    with open(path) as fh:
        return [json.loads(l) for l in fh]


def causal_score(row):
    return 1.0 - row["probs"][VARIANTS[row["variant"]][1]]


def prf_at(y, s, t):
    tp = sum(1 for a, b in zip(y, s) if a and b >= t)
    fp = sum(1 for a, b in zip(y, s) if not a and b >= t)
    fn = sum(1 for a, b in zip(y, s) if a and b < t)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def best_threshold(y, s):
    import numpy as np
    from sklearn.metrics import precision_recall_curve
    p, r, t = precision_recall_curve(y, s)
    f = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0)[:-1]
    k = int(f.argmax())
    return float(t[k]), float(f[k])


def binary_row(rows, dev_rows=None):
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = [r["gold"] for r in rows]
    s = [causal_score(r) for r in rows]
    prev = sum(y) / len(y)
    argmax = [int(max(r["probs"], key=r["probs"].get) != VARIANTS[r["variant"]][1]) for r in rows]
    pa, ra, fa = prf_at(y, argmax, 1)
    out = {"pairs": len(y), "gold": sum(y), "prev": prev,
           "auroc": roc_auc_score(y, s), "ap": average_precision_score(y, s),
           "argmax_p": pa, "argmax_r": ra, "argmax_f1": fa, "pred_pos": sum(argmax) / len(y),
           "oracle_t": None, "oracle_f1": None, "dev_t": None, "dev_p": None, "dev_r": None, "dev_f1": None,
           "allpos_f1": 2 * prev / (1 + prev)}
    out["oracle_t"], out["oracle_f1"] = best_threshold(y, s)
    if dev_rows:
        t, _ = best_threshold([r["gold"] for r in dev_rows], [causal_score(r) for r in dev_rows])
        out["dev_t"] = t
        out["dev_p"], out["dev_r"], out["dev_f1"] = prf_at(y, s, t)
    return out


def direction_row(rows):
    """On gold causal pairs: does P(event_1 -> event_2) vs P(event_2 -> event_1) pick the gold
    direction? Baseline: always text order (the earlier mention is the cause)."""
    fwd, bwd = VARIANTS["dir3"][2:4]
    g = [r for r in rows if r["gold"]]
    acc = sum((r["probs"][fwd] >= r["probs"][bwd]) == (r["gold_dir"] == "12") for r in g) / len(g)
    text_order = sum(r["gold_dir"] == "12" for r in g) / len(g)
    says_fwd = sum(r["probs"][fwd] >= r["probs"][bwd] for r in g) / len(g)
    return {"gold_pairs": len(g), "dir_acc": acc, "text_order_acc": text_order, "pred_fwd": says_fwd}


def cmd_score(args):
    """args.pairs: eval=dev jsonl pairs (dev may be '-'). Prints markdown tables."""
    cols = ["pairs", "gold", "prev", "auroc", "ap", "argmax_p", "argmax_r", "argmax_f1", "pred_pos",
            "dev_t", "dev_p", "dev_r", "dev_f1", "oracle_f1", "allpos_f1"]
    print("| run | " + " | ".join(cols) + " |")
    print("|---|" + "---|" * len(cols))
    for spec in args.pairs:
        name, _, rest = spec.partition(":")
        ev, _, dev = rest.partition("=")
        m = binary_row(load_rows(ev), load_rows(dev) if dev and dev != "-" else None)
        fmt = lambda v: "—" if v is None else (f"{v:,}" if isinstance(v, int) else f"{v:.3f}")
        print(f"| {name} | " + " | ".join(fmt(m[c]) for c in cols) + " |")
    if args.direction:
        print("\n| run | gold_pairs | dir_acc | text_order_acc | pred_fwd |\n|---|---|---|---|---|")
        for spec in args.direction:
            name, _, path = spec.partition(":")
            m = direction_row(load_rows(path))
            print(f"| {name} | {m['gold_pairs']:,} | {m['dir_acc']:.3f} | {m['text_order_acc']:.3f} | "
                  f"{m['pred_fwd']:.3f} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--backend", choices=["laya", "jev"], required=True)
    p.add_argument("--checkpoint", choices=["english", "multilingual", "typed-decisions"], default="english")
    p.add_argument("--jev-model", default="typesafe/jev-1.13")
    p.add_argument("--concurrency", type=int, default=4, help="Jev: parallel requests")
    p.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    p.add_argument("--dataset", choices=sorted(REPOS), required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--max-docs", type=int, default=0, help="0 = every document of the split")
    p.add_argument("--doc-seed", type=int, default=0)
    p.add_argument("--gold-only", action="store_true", help="only gold causal pairs (direction study)")
    p.add_argument("--limit", type=int, default=0, help="first N pairs only (0 = all)")
    p.add_argument("--batch-size", type=int, default=64, help="pairs per backend call / jsonl flush")
    p.add_argument("--forward-batch", type=int, default=0,
                   help="Laya: split each call into length-sorted forward passes of this size (0 = one pass)")
    p.add_argument("--threads", type=int, default=24)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("pairs", nargs="+", help="name:eval.jsonl=dev.jsonl ('-' for no dev)")
    s.add_argument("--direction", nargs="*", default=[], help="name:dir3.jsonl")
    args = ap.parse_args()
    {"predict": cmd_predict, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
