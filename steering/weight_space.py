"""Rank-one additive MLP edits from Concept Country, §3.3 (2026-11-08).

Delta W = alpha * mean_output_norm * outer(d, u) / dot(u, u).
On the quantized base we evaluate W4(x) + Delta W(x) in a persistent hook.
This is a full-precision sidecar, NOT a claim about re-quantized merged weights.
All token positions receive the edit, as a real weight edit would.
"""
from __future__ import annotations

import hashlib
import json
import random
from typing import Any

import numpy as np

LAYERS = tuple(range(32, 49, 2))
SEED = 20261110


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def unit(vector):
    vector = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(vector)
    if vector.ndim != 1 or not np.isfinite(vector).all() or norm < 1e-10:
        raise ValueError("direction must be a finite nonzero vector")
    return (vector / norm).astype(np.float32)


def orthogonal_control(direction, seed: int):
    d = unit(direction).astype(np.float64)
    r = np.random.default_rng(seed).normal(size=d.shape)
    r -= d * (r @ d) / (d @ d)
    return unit(r)


def right_factor(mean_input, mean_output_norm: float):
    u = np.asarray(mean_input, dtype=np.float64)
    if u.ndim != 1 or not np.isfinite(u).all() or u @ u < 1e-12:
        raise ValueError("mean writer input must be finite and nonzero")
    if not np.isfinite(mean_output_norm) or mean_output_norm <= 0:
        raise ValueError("mean output norm must be positive")
    return (mean_output_norm * u / (u @ u)).astype(np.float32)


def make_split(suite, prompt, excluded, screen_size=20, confirmation_size=20):
    """Split by document BEFORE reading traces or outcomes; reserve all demo docs for fit."""
    excluded = set(excluded)
    demos = {d["id"] for d in prompt["demos"]}
    train = sorted(d for d, row in suite.items() if d not in excluded and row["pair_list"])
    eligible = sorted(set(train) - demos)
    random.Random(SEED).shuffle(eligible)
    if len(eligible) <= screen_size + confirmation_size:
        raise ValueError("insufficient non-demonstration training documents")
    screen = sorted(eligible[:screen_size])
    confirmation = sorted(eligible[screen_size:screen_size + confirmation_size])
    fit = sorted(set(train) - set(screen) - set(confirmation))
    groups = {"fit": fit, "screen": screen, "confirmation": confirmation,
              "excluded_historical": sorted(excluded)}
    for name, ids in groups.items():
        for other, values in groups.items():
            if name != other and set(ids) & set(values):
                raise ValueError(f"split overlap: {name}/{other}")
    if (set(screen) | set(confirmation)) & demos:
        raise ValueError("demonstration leakage")
    return {"seed": SEED, **groups, "demonstrations": sorted(demos)}


def final_pairs(row):
    """Parse final answer ONLY, never arrays embedded in the thinking trace.

    None means a malformed/truncated answer, including malformed items that the
    legacy permissive parser silently drops. Empty set is a valid no-link answer.
    """
    from eci.labels import normalize

    if row.get("finish_reason") != "stop" or "</think>" not in row.get("text", ""):
        return None
    answer = row["text"].split("</think>", 1)[1].strip()
    if answer.startswith("```") and answer.endswith("```"):
        answer = answer.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        values = json.loads(answer)
    except (ValueError, TypeError):
        return None
    if not isinstance(values, list):
        return None
    pairs = set()
    for item in values:
        if not isinstance(item, dict) or not isinstance(item.get("pair"), str):
            return None
        ids = [v.strip() for v in item["pair"].split(",")]
        if len(ids) != 2 or not all(ids) or ids[0] == ids[1]:
            return None
        if not isinstance(item.get("label"), str):
            return None
        label = normalize(item["label"])
        if label not in ("causes", "causedby", "norel"):
            return None
        if label != "norel":
            pairs.add(tuple(ids))
    return pairs


class RankOneWriter:
    """Device buffers keep arm switches visible inside captured decode graphs."""

    def __init__(self, module, device="cuda", dtype=None):
        import torch

        self.right = torch.zeros(module.input_size, device=device, dtype=torch.float32)
        self.left = torch.zeros(module.output_size, device=device, dtype=torch.float32)
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        import torch

        y = output[0] if isinstance(output, tuple) else output
        # Coefficient in FP32; no re-quantization and no input-dependent clipping.
        # Bound the prefill workspace on a 24-GB card. Casting a whole long MLP
        # input to FP32 would otherwise allocate multiple extra gigabytes.
        x = inputs[0].reshape(-1, self.right.numel())
        coefficient = torch.cat([x[start:start + 256].float() @ self.right
                                 for start in range(0, x.shape[0], 256)]).reshape(y.shape[:-1])
        edited = y + (coefficient.unsqueeze(-1) * self.left).to(y.dtype)
        return (edited, *output[1:]) if isinstance(output, tuple) else edited

    def set(self, direction, right, alpha):
        import torch

        if not np.isfinite(alpha):
            raise ValueError("non-finite alpha")
        d = unit(direction)
        r = np.asarray(right, dtype=np.float32)
        if d.shape != tuple(self.left.shape) or r.shape != tuple(self.right.shape):
            raise ValueError("rank-one factor shape mismatch")
        if not np.isfinite(r).all():
            raise ValueError("non-finite right factor")
        with torch.inference_mode():
            self.right.copy_(torch.as_tensor(r, device=self.right.device))
            self.left.copy_(torch.as_tensor(d * alpha, device=self.left.device))

    def clear(self):
        self.left.zero_()


_WRITERS = {}


def install_writers(layers=LAYERS):
    """Install before engine initialization/CUDA capture; one GPU, tensor parallel = 1."""
    from vllm.model_executor.layers.linear import RowParallelLinear

    if _WRITERS:
        raise RuntimeError("writers already installed in this process")
    original = RowParallelLinear.__init__

    def patched(self, *args, **kwargs):
        original(self, *args, **kwargs)
        prefix = str(kwargs.get("prefix", ""))
        for layer in layers:
            if prefix.endswith(f"layers.{layer}.mlp.down_proj"):
                _WRITERS[layer] = RankOneWriter(self)

    RowParallelLinear.__init__ = patched
    return _WRITERS


def condition_plan():
    arms = [{"name": "base", "family": None, "alpha": 0.0}]
    for family in ("cc", "recall"):
        for alpha in (0.05, 0.10):
            arms.append({"name": f"{family}_{alpha:.2f}", "family": family, "alpha": alpha})
        for seed in (0, 1):
            arms.append({"name": f"{family}_random{seed}_0.10", "family": family,
                         "random_seed": seed, "alpha": 0.10})
    return [{**a, "budget": 8192, "answer_tokens": 4096, "temperature": 0.0,
             "layers": list(LAYERS)} for a in arms]


def counts(predicted, gold):
    return {"tp": len(predicted & gold), "fp": len(predicted - gold),
            "fn": len(gold - predicted)}


def metrics(per_doc):
    tp, fp, fn = (sum(row[k] for row in per_doc.values()) for k in ("tp", "fp", "fn"))
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0}


def paired_interval(a, b, draws=10000):
    """Document-block bootstrap of micro F1, b minus a, in percentage points."""
    if not a or set(a) != set(b):
        raise ValueError("paired comparisons require identical nonempty document sets")
    ids = sorted(a)
    arrays = [np.array([[m[d][k] for k in ("tp", "fp", "fn")] for d in ids]) for m in (a, b)]
    indices = np.random.default_rng(SEED).integers(0, len(ids), (draws, len(ids)))
    scores = []
    for array in arrays:
        total = array[indices].sum(1)
        den = 2 * total[:, 0] + total[:, 1] + total[:, 2]
        scores.append(np.divide(200 * total[:, 0], den, out=np.zeros(draws), where=den > 0))
    lo, hi = np.quantile(scores[1] - scores[0], [0.025, 0.975])
    return [100 * (metrics(b)["f1"] - metrics(a)["f1"]), float(lo), float(hi)]
