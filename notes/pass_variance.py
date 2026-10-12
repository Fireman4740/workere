#!/usr/bin/env python3
"""Run-to-run spread of binary F1, from the per-pass predictions rebuilt out of the traces.

Novita runs only, one config (deepseek-v4-flash, 1 demo/pass from a pool of 20, CoT demos,
coref-aware prompt on ESC/MAVEN, syntax off):

  CTB    a43655dd (7 passes)  525746e4 (1 pass)  1dd870a8 (1 pass, crashed at 159 docs)
  ESC    e77431d6 (7)         03dd2a75 (1)
  MAVEN  dfe087fb (7)         1ac866ca (7, only its Novita-priced calls)   a0293913 (1)

Pass r of every run of a document sends a byte-identical request when the demo slot r matches
(checked on the traces), so each document carries 7 demo SLOTS, each with 1-3 decoding
REPLICATES. Two kinds of run-to-run noise follow:

  new seed   the demos are redrawn and the decoding repeated. Unbiased for N <= 3 passes: per
             document, (L(S) - L(S'))^2 / 2 averaged over disjoint slot sets S, S' (and over the
             replicate chosen for each slot), summed over documents. L is the document's
             linearised contribution to the corpus F1 (delta method; TP and FP totals, gold
             fixed). N = 7 needs 14 slots, so it comes from a bootstrap over slots, whose bias is
             measured against the unbiased estimator at N = 1..3.
  same seed  the run repeated as is: identical requests, decoding only. Slot 0 (= the 1-pass
             config) on every dataset; slots 0..6 on the MAVEN documents whose 1ac866ca passes
             were all served by Novita.

A pair is predicted at threshold t when >= t of the N passes label it causes/causedby (the
analyze.py sweep convention, ordered pairs, gold as stored in both directions).
"""
from __future__ import annotations

import glob
import hashlib
import itertools
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eci import parse, voting  # noqa: E402
from eci.labels import CAUSAL  # noqa: E402

LOGS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "runs")
DATASETS = {"ctb": ("a43655dd", ["525746e4", "1dd870a8"], None),
            "esc": ("e77431d6", ["03dd2a75"], None),
            "maven": ("dfe087fb", ["a0293913"], "1ac866ca")}
SCOPES = ("all", "intra", "inter")
NOVITA_PER_M = 0.28


def rundir(r):
    return glob.glob(f"{LOGS}/*{r}")[0]


def read_passes(r):
    """{(doc, pass): {"pairs": causal ordered pairs, "req": request hash, "novita": bool}}."""
    d = rundir(r)
    out = {}
    for line in open(f"{d}/{os.path.basename(d)}.traces.jsonl", encoding="utf-8"):
        x = json.loads(line)
        if x["role"] != "inference" or not x["ok"]:
            continue
        msgs = x["messages"]
        req = json.dumps([(m["id"][-1], m["kwargs"].get("content")) for m in msgs[:-1]], ensure_ascii=False)
        resp = msgs[-1]["kwargs"]
        tu = (resp.get("response_metadata") or {}).get("token_usage") or {}
        cost = (tu.get("cost_details") or {}).get("upstream_inference_completions_cost")
        ct = tu.get("completion_tokens")
        labels = {}
        for s, lab, t in parse.parse_triples(resp["content"]):
            labels[(s, t)] = lab                      # last label wins, as in voting.aggregate
        out[(x["doc_id"], x["pass"])] = {
            "triples": parse.parse_triples(resp["content"]),
            "pairs": frozenset(p for p, lab in labels.items() if lab in CAUSAL),
            "req": hashlib.sha256(req.encode()).hexdigest(),
            "novita": cost is not None and bool(ct) and abs(cost / ct * 1e6 - NOVITA_PER_M) < 1e-3}
    return out


def check_reconstruction(r, passes):
    """The rebuilt passes re-aggregate to the run's own checkpointed vote tallies."""
    d = rundir(r)
    n = ok = 0
    for line in open(f"{d}/docs.jsonl", encoding="utf-8"):
        res = json.loads(line)
        if res["failure"]:
            continue
        per = [passes[k]["triples"] for k in sorted((k for k in passes if k[0] == res["id"]), key=lambda k: k[1])]
        stats = voting.aggregate(per, "norel", random.Random(0))[1] if len(per) > 1 else voting.single_pass_stats(per[0])
        n += 1
        ok += {k: v["vote_counts"] for k, v in stats.items()} == {k: v["vote_counts"] for k, v in res["pair_stats"].items()}
    return ok, n


class Doc:
    """One document: pair universe, gold and scope masks, and slot -> replicate pass vectors."""

    def __init__(self, res, slots):
        gold = {(s, t) for s, lab, t in res["gold_triples"] if lab in CAUSAL}
        sent = res["mention_sentence"]
        universe = sorted(gold | set().union(*(p for reps in slots for p in reps)))
        idx = {p: i for i, p in enumerate(universe)}
        self.id = res["id"]
        self.gold = np.array([p in gold for p in universe], bool)
        rel = ["unknown" if sent.get(a) is None or sent.get(b) is None else
               "intra" if sent[a] == sent[b] else "inter" for a, b in universe]
        self.scope = np.stack([np.ones(len(universe), bool), np.array([x == "intra" for x in rel], bool),
                               np.array([x == "inter" for x in rel], bool)])          # (3, U)
        self.n_gold = (self.scope & self.gold).sum(1)                           # (3,)
        self.reps = []                                                          # slot -> (R, U)
        for reps in slots:
            m = np.zeros((len(reps), len(universe)), np.int16)
            for j, p in enumerate(reps):
                m[j, [idx[x] for x in p]] = 1
            self.reps.append(m)

    def counts(self, votes, n):
        """votes (B, U) -> tp, fp of shape (B, n thresholds, 3 scopes)."""
        pred = votes[:, None, :] >= np.arange(1, n + 1)[None, :, None]           # (B, T, U)
        sc = self.scope.T.astype(np.int32)                                      # (U, 3)
        return (pred & self.gold).astype(np.int32) @ sc, (pred & ~self.gold).astype(np.int32) @ sc


def load(ds):
    main, singles, twin = DATASETS[ds]
    runs = [main] + singles + ([twin] if twin else [])
    P = {r: read_passes(r) for r in runs}
    recon = {r: check_reconstruction(r, P[r]) for r in runs}
    docs, mismatch, baidu = [], 0, 0
    for line in open(f"{rundir(main)}/docs.jsonl", encoding="utf-8"):
        res = json.loads(line)
        slots, twin_ok = [], True
        for s in range(7):
            ref = P[main][(res["id"], s)]
            reps = [ref]
            extra = [P[r].get((res["id"], 0)) for r in singles] if s == 0 else []
            if twin:
                t = P[twin][(res["id"], s)]
                if t["novita"]:
                    extra.append(t)
                else:
                    twin_ok, baidu = False, baidu + 1
            for e in extra:
                if e is None:
                    continue
                mismatch += e["req"] != ref["req"]
                reps.append(e)
            slots.append([x["pairs"] for x in reps])
        doc = Doc(res, slots)
        doc.twin_complete = bool(twin) and twin_ok
        if len(doc.gold):            # no gold and no pass predicts anything: scores nothing
            docs.append(doc)
    return docs, recon, mismatch, baidu


# ── estimators ────────────────────────────────────────────────────────────────

def f1(tp, fp, g):
    d = tp + fp + g
    return np.where(d > 0, 2 * tp / np.where(d > 0, d, 1), 0.0)


def grad(TP, FP, G):
    """dF/dTP, dF/dFP with FN = G - TP; F = 2TP / (TP + FP + G)."""
    D = TP + FP + G
    return 2 * (FP + G) / D ** 2, -2 * TP / D ** 2


def version_matrix(doc, rng, n_versions):
    """(V, 7, U) pass vectors, one replicate per slot per version; exhaustive when small."""
    sizes = [len(r) for r in doc.reps]
    combos = list(itertools.product(*[range(k) for k in sizes]))
    if len(combos) > n_versions:
        combos = [tuple(rng.integers(k) for k in sizes) for _ in range(n_versions)]
    return np.stack([np.stack([doc.reps[s][c[s]] for s in range(7)]) for c in combos])


def new_seed(docs, N, rng, n_versions=32):
    """Expected F1 and unbiased variance of an N-pass run under a fresh demo draw (N <= 3)."""
    subsets = list(itertools.combinations(range(7), N))
    sub_idx = {s: i for i, s in enumerate(subsets)}
    pairs = [(sub_idx[a], sub_idx[b]) for a in subsets for b in subsets if not set(a) & set(b)]
    M = np.zeros((len(subsets), 7), np.int16)
    for i, s in enumerate(subsets):
        M[i, list(s)] = 1
    per_doc = []
    for d in docs:
        V = version_matrix(d, rng, n_versions)                      # (V, 7, U)
        votes = np.einsum("ks,vsu->vku", M, V).reshape(-1, V.shape[2])
        tp, fp = d.counts(votes, N)
        per_doc.append((tp.reshape(len(V), len(subsets), N, 3), fp.reshape(len(V), len(subsets), N, 3)))
    G = sum(d.n_gold for d in docs)[None, :]
    TP = sum(tp.mean((0, 1)) for tp, _ in per_doc)
    FP = sum(fp.mean((0, 1)) for _, fp in per_doc)
    if not pairs:
        return f1(TP, FP, G), np.full(TP.shape, np.nan), None
    a, b = grad(TP, FP, G)
    ia, ib = np.array(pairs).T
    var, contrib = 0.0, []
    for tp, fp in per_doc:
        L = a * tp + b * fp                                         # (V, K, T, 3)
        v = ((L[:, ia] - L[:, ib]) ** 2 / 2).mean((0, 1))
        contrib.append(v)
        var = var + v
    return f1(TP, FP, G), var, np.array(contrib)


def direct_new_seed(docs, N, rng, n_draws=400):
    """Same quantity without linearisation: corpus F1 of two disjoint N-slot runs."""
    G = sum(d.n_gold for d in docs)[None, :]
    TPa = FPa = TPb = FPb = 0
    for d in docs:
        perm = np.argsort(rng.random((n_draws, 7)), 1)
        rep = [rng.integers(len(r), size=n_draws) for r in d.reps]
        X = np.stack([d.reps[s][rep[s]] for s in range(7)], 1)      # (draws, 7, U)
        A = np.take_along_axis(X, perm[:, :N, None], 1).sum(1)
        B = np.take_along_axis(X, perm[:, N:2 * N, None], 1).sum(1)
        ta, fa = d.counts(A, N)
        tb, fb = d.counts(B, N)
        TPa, FPa, TPb, FPb = TPa + ta, FPa + fa, TPb + tb, FPb + fb
    return ((f1(TPa, FPa, G) - f1(TPb, FPb, G)) ** 2 / 2).mean(0)


def bootstrap_new_seed(docs, N, rng, B=2000):
    """sd of corpus F1 when each document's N passes are drawn with replacement from its 7 slots."""
    G = sum(d.n_gold for d in docs)[None, :]
    TP = FP = 0
    for d in docs:
        slot = rng.integers(7, size=(B, N))
        sizes = np.array([len(r) for r in d.reps])
        offs = np.cumsum(np.r_[0, sizes])[:-1]
        stacked = np.concatenate(d.reps)                             # (sum R, U)
        rep = (rng.random((B, N)) * sizes[slot]).astype(int)
        votes = stacked[offs[slot] + rep].sum(1)
        tp, fp = d.counts(votes, N)
        TP, FP = TP + tp, FP + fp
    F = f1(TP, FP, G)
    return F.mean(0), F.var(0, ddof=1)


def same_seed(docs, slots, rng, max_choices=256):
    """Expected F1 and unbiased variance of re-running the given slots with identical requests.
    Per document: two runs that use different replicates of every slot."""
    N = len(slots)
    usable = [d for d in docs if all(len(d.reps[s]) >= 2 for s in slots)]
    per_doc = []
    for d in usable:
        choices = list(itertools.product(*[list(itertools.permutations(range(len(d.reps[s])), 2)) for s in slots]))
        if len(choices) > max_choices:
            choices = [choices[i] for i in rng.choice(len(choices), max_choices, replace=False)]
        A = np.stack([sum(d.reps[s][c[k][0]] for k, s in enumerate(slots)) for c in choices])
        Bv = np.stack([sum(d.reps[s][c[k][1]] for k, s in enumerate(slots)) for c in choices])
        ta, fa = d.counts(A, N)
        tb, fb = d.counts(Bv, N)
        per_doc.append((ta, fa, tb, fb))
    G = sum(d.n_gold for d in usable)[None, :]
    TP = sum((ta.mean(0) + tb.mean(0)) / 2 for ta, _, tb, _ in per_doc)
    FP = sum((fa.mean(0) + fb.mean(0)) / 2 for _, fa, _, fb in per_doc)
    a, b = grad(TP, FP, G)
    contrib = np.array([(((a * ta + b * fa) - (a * tb + b * fb)) ** 2 / 2).mean(0) for ta, fa, tb, fb in per_doc])
    return f1(TP, FP, G), contrib.sum(0), contrib


def doc_boot_sd(contrib, rng, B=2000):
    """Spread of the summed variance when documents are resampled: uncertainty of the estimate."""
    n = len(contrib)
    tot = np.stack([contrib[rng.integers(n, size=n)].sum(0) for _ in range(B)])
    return np.sqrt(np.percentile(tot, 2.5, 0)), np.sqrt(np.percentile(tot, 97.5, 0))


def realised(docs, pick):
    """Corpus F1 (T, 3) of one actual run: pick(doc) -> vote vector, scored at t = 1..N."""
    TP = FP = 0
    N = None
    for d in docs:
        votes, N = pick(d)
        tp, fp = d.counts(votes[None], N)
        TP, FP = TP + tp[0], FP + fp[0]
    return f1(TP, FP, sum(d.n_gold for d in docs)[None, :])


def summarise(F, var, contrib, rng):
    out = {"F": F.tolist(), "sd": np.sqrt(var).tolist()}
    if contrib is not None:
        lo, hi = doc_boot_sd(contrib, rng)
        out.update(sd_lo=lo.tolist(), sd_hi=hi.tolist())
    return out


def main():
    rng = np.random.default_rng(0)
    out = {}
    for ds in DATASETS:
        docs, recon, mismatch, baidu = load(ds)
        r = {"docs": len(docs), "recon": recon, "request_mismatch": mismatch, "twin_baidu_passes": baidu,
             "replicates": {s: {int(k): int(v) for k, v in zip(*np.unique([len(d.reps[s]) for d in docs], return_counts=True))}
                            for s in range(7)}}
        print(f"\n=== {ds}: {len(docs)} scored docs; reconstruction {recon}; request mismatches {mismatch}; "
              f"twin Baidu passes {baidu}", flush=True)
        # the 7 single-pass runs hidden in the 7-pass run (first replicate of each slot)
        r["slot_runs"] = [realised(docs, lambda d, s=s: (d.reps[s][0], 1))[0].tolist() for s in range(7)]
        r["main_run"] = realised(docs, lambda d: (sum(d.reps[s][0] for s in range(7)), 7)).tolist()
        for N in (1, 2, 3, 7):
            F, var, contrib = new_seed(docs, N, rng)
            r[f"new_N{N}"] = summarise(F, var, contrib, rng)
            bm, bv = bootstrap_new_seed(docs, N, rng)
            r[f"new_N{N}"].update(sd_boot=np.sqrt(bv).tolist(), F_boot=bm.tolist())
            if N <= 3:
                r[f"new_N{N}"]["sd_direct"] = np.sqrt(direct_new_seed(docs, N, rng)).tolist()
            print(f"N={N} F={np.round(100 * F[:, 0], 2)} sd={np.round(100 * np.sqrt(var[:, 0]), 2)} "
                  f"boot={np.round(100 * np.sqrt(bv[:, 0]), 2)}", flush=True)
        F, var, contrib = same_seed(docs, [0], rng)
        r["same_N1"] = {**summarise(F, var, contrib, rng), "docs": len(contrib)}
        print(f"same-seed N=1 ({len(contrib)} docs) F={np.round(100 * F[:, 0], 2)} sd={np.round(100 * np.sqrt(var[:, 0]), 2)}", flush=True)
        if DATASETS[ds][2]:
            sub = [d for d in docs if d.twin_complete]
            r["twin_docs"] = len(sub)
            for N in (1, 3, 7):
                F, var, contrib = same_seed(sub, list(range(N)), rng)
                r[f"twin_same_N{N}"] = summarise(F, var, contrib, rng)
                F, var, contrib = new_seed(sub, N, rng)
                r[f"twin_new_N{N}"] = summarise(F, var, contrib, rng)
                bm, bv = bootstrap_new_seed(sub, N, rng)
                r[f"twin_boot_N{N}"] = {"F": bm.tolist(), "sd": np.sqrt(bv).tolist()}
                print(f"[twin {len(sub)} docs] N={N} same-seed sd={np.round(100 * np.array(r[f'twin_same_N{N}']['sd'])[:, 0], 2)} "
                      f"new-seed sd={np.round(100 * np.sqrt(var[:, 0]), 2)} boot={np.round(100 * np.sqrt(bv[:, 0]), 2)}", flush=True)
            r["twin_run_dfe087fb"] = realised(sub, lambda d: (sum(d.reps[s][0] for s in range(7)), 7)).tolist()
            r["twin_run_1ac866ca"] = realised(sub, lambda d: (sum(d.reps[s][-1] for s in range(7)), 7)).tolist()
        out[ds] = r
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pass_variance.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
