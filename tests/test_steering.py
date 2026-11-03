"""CPU-side checks for the N13 CTB50 steering pilot (no GPU, no vLLM)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "steering"))

from steering.build_self_contrast import build_pairs, render_answer  # noqa: E402
from steering.compare import micro_f1, paired_delta  # noqa: E402
from steering.extract_prompts import split_trace, wanted  # noqa: E402
from steering.make_controls import random_unit  # noqa: E402
from steering.pilot_conditions import build_conditions  # noqa: E402

GOLD = [("ei1", "causes", "ei2"), ("ei2", "causedby", "ei1")]


def answer(*pairs):
    return json.dumps([{"pair": f"{a},{b}", "label": lab} for a, lab, b in pairs])


def test_render_answer_roundtrips_gold():
    assert json.loads(render_answer(GOLD)) == [
        {"pair": "ei1,ei2", "label": "causes"},
        {"pair": "ei2,ei1", "label": "causedby"},
    ]


def test_build_pairs_keeps_worst_wrong_pass_only():
    rows = [
        {"doc_id": "d", "pass": 0, "messages": [], "answer": answer(*GOLD)},
        {"doc_id": "d", "pass": 1, "messages": [], "answer": answer(GOLD[0])},
        {"doc_id": "d", "pass": 2, "messages": [], "answer": "[]"},
    ]
    pairs = build_pairs(rows, {"d": GOLD}, eval_ids=set())
    assert len(pairs) == 1
    assert pairs[0]["pass"] == 2 and pairs[0]["bad_f1"] == 0.0
    assert pairs[0]["good_answer"] == render_answer(GOLD)


def test_build_pairs_skips_perfect_unparsable_and_goldless():
    rows = [
        {"doc_id": "perfect", "pass": 0, "messages": [], "answer": answer(*GOLD)},
        {"doc_id": "garbage", "pass": 0, "messages": [], "answer": "no json here"},
        {"doc_id": "nogold", "pass": 0, "messages": [], "answer": answer(GOLD[0])},
    ]
    gold = {"perfect": GOLD, "garbage": GOLD, "nogold": []}
    assert build_pairs(rows, gold, eval_ids=set()) == []


def test_build_pairs_refuses_evaluation_documents():
    rows = [{"doc_id": "d", "pass": 0, "messages": [], "answer": "[]"}]
    with pytest.raises(ValueError, match="evaluation documents"):
        build_pairs(rows, {"d": GOLD}, eval_ids={"d"})


def test_trace_split_and_mode_filter():
    def msg(kind, content):
        return {"id": ["x", kind], "kwargs": {"content": content}}

    row = {"messages": [msg("SystemMessage", "s"), msg("HumanMessage", "u"), msg("AIMessage", "a")]}
    prompt, final = split_trace(row)
    assert [m["role"] for m in prompt] == ["system", "user"] and final == "a"
    assert wanted("x", {"x"}, "eval") and not wanted("x", {"x"}, "train")
    assert wanted("y", {"x"}, "train") and not wanted("y", {"x"}, "eval")


def test_micro_f1_and_paired_delta_detects_clear_gain():
    control = {f"d{i}": {"tp": 1, "fp": 1, "fn": 1} for i in range(20)}
    better = {f"d{i}": {"tp": 2, "fp": 0, "fn": 0} for i in range(20)}
    assert micro_f1(list(control.values())) == pytest.approx(50.0)
    delta, lo, hi = paired_delta(better, control, draws=500)
    assert delta == pytest.approx(50.0) and lo > 0
    same, lo, hi = paired_delta(control, control, draws=500)
    assert same == 0 and lo == 0 == hi


def test_random_controls_are_unit_seeded_and_distinct():
    a, b = random_unit(64, 1), random_unit(64, 2)
    assert np.linalg.norm(a) == pytest.approx(1.0, abs=1e-5)
    assert np.array_equal(a, random_unit(64, 1)) and not np.array_equal(a, b)


def test_pilot_order_and_stages():
    conditions = build_conditions("d", layer=40, module="down_proj", alpha=0.15, n_random=2)
    stage1 = [c["name"] for c in conditions if c["stage"] == 1]
    assert stage1 == ["control", "hint_pos", "demo_pos", "self_pos"]
    assert conditions[0]["direction"] is None and conditions[0]["alpha"] == 0.0
    names = {c["name"] for c in conditions if c["stage"] == 2}
    assert {"hint_neg", "demo_rand1", "self_neg"} <= names


def test_steerer_dose_is_a_buffer_and_control_is_exact():
    torch = pytest.importorskip("torch")
    from offline import Steerer

    linear = torch.nn.Linear(4, 4, bias=False)
    torch.nn.init.zeros_(linear.weight)
    steerer = Steerer(linear, hidden_size=4, positions="last", device="cpu")
    assert linear(torch.zeros(3, 4)).abs().sum() == 0
    direction = torch.zeros(4)
    direction[0] = 1.0
    steerer.set(direction, alpha=0.5, ref_norm=10.0)
    out = linear(torch.zeros(3, 4))
    assert out[-1, 0] == 5.0 and out[:-1].abs().sum() == 0
    assert linear(torch.zeros(1, 4))[0, 0] == 5.0
    steerer.clear()
    assert linear(torch.zeros(3, 4)).abs().sum() == 0
    with pytest.raises(ValueError):
        Steerer(linear, hidden_size=4, positions="middle", device="cpu")
