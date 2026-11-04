"""CPU checks for the E1-E5 reasoning-mechanism experiments (no GPU, no vLLM)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "steering"))

import ctb  # noqa: E402
from early_exit import answer_key, majority  # noqa: E402
from probe import auroc, best_threshold, end_positions, pair_block  # noqa: E402
from reflection import label, sentence_windows, sentences  # noqa: E402


def test_projector_removes_only_positive_component():
    torch = pytest.importorskip("torch")
    from interventions import Projector

    linear = torch.nn.Linear(3, 3, bias=False)
    torch.nn.init.eye_(linear.weight)
    projector = Projector(linear, 3, device="cpu")
    projector.set(torch.tensor([1.0, 0.0, 0.0]), strength=1.0)
    out = linear(torch.tensor([[2.0, 1.0, 0.0], [-2.0, 1.0, 0.0]]))
    assert torch.allclose(out, torch.tensor([[0.0, 1.0, 0.0], [-2.0, 1.0, 0.0]]))
    projector.clear()
    assert torch.allclose(linear(torch.ones(1, 3)), torch.ones(1, 3))


def test_head_projector_equals_weight_edit():
    torch = pytest.importorskip("torch")
    from interventions import HeadProjector

    torch.manual_seed(0)
    linear = torch.nn.Linear(4, 3, bias=False)
    edit = HeadProjector(linear, 4, 3, device="cpu")
    d = torch.tensor([0.0, 1.0, 0.0])
    u = linear.weight.detach().T @ d
    u[2:] = 0
    edit.set(u, d)
    x = torch.randn(5, 4)
    edited_weight = linear.weight.detach().clone()
    edited_weight[:, :2] -= torch.outer(d, d @ edited_weight[:, :2])
    assert torch.allclose(linear(x), x @ edited_weight.T, atol=1e-5)


def test_hesitation_labels_and_sentence_offsets():
    text = "The pair is causal. But wait, maybe not.\nShould we include ei3? It is anchored by because."
    parts = sentences(text)
    assert [text[s:s + len(t)] == t for s, t in parts] == [True] * len(parts)
    assert [label(t) for _, t in parts] == ["d", "h", "h", "d"]


class FakeTokenizer:
    """One token per character, offsets as a fast tokenizer would return them."""

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        return {"input_ids": list(range(len(text))), "offset_mapping": [(i, i + 1) for i in range(len(text))]}


def test_sentence_windows_start_at_sentence_onsets():
    prompt, thinking = "PROMPT ", "First claim holds here. But wait, check it."
    full, windows = sentence_windows(FakeTokenizer(), prompt, thinking, 5, "s", 10_000)
    assert full == prompt + thinking
    onsets = {cls: full[w[0]:w[0] + 4] for cls, w in windows}
    assert onsets == {"d": "Firs", "h": "But "}


def test_pair_block_positions_hit_line_ends():
    doc = {"surfaces": {"ei1": "rose", "ei2": "fell", "ei3": "said"},
           "pair_list": [["ei2", "ei1"], ["ei1", "ei2"], ["ei1", "ei3"]]}
    block, pairs, ends = pair_block(doc)
    assert pairs == [("ei1", "ei2"), ("ei1", "ei3")]
    head = "HEAD"
    positions = end_positions(FakeTokenizer(), head + block, [len(head) + e for e in ends])
    assert [(head + block)[p] for p in positions] == [":", ":"]


def test_early_exit_majority_needs_two_agreeing_answers():
    a = '</think>\n[{"pair": "ei1,ei2", "label": "causes"}, {"pair": "ei2,ei1", "label": "causedby"}]'
    b = '</think>\n[]'
    assert answer_key("no close") is None
    key, text = majority([a, b, a])
    assert key == frozenset({("ei1", "ei2"), ("ei2", "ei1")}) and text == a
    assert majority([a, b, "still thinking"]) == (None, None)


def test_probe_threshold_and_auroc():
    y = np.array([False, False, True, True])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    assert auroc(s, y) == 1.0
    f1, threshold = best_threshold(s, y)
    assert f1 == 1.0 and 0.2 < threshold <= 0.8


def test_training_split_excludes_evaluation_documents():
    suite, prompt = ctb.load_suite(), ctb.load_prompt()
    ids = ctb.train_ids(suite, prompt, set(ctb.eval_ids()))
    assert not set(ids) & set(ctb.eval_ids())
    for doc_id in ids[:20]:
        messages = ctb.messages_for(suite[doc_id], prompt)
        assert messages[1]["content"] != messages[3]["content"]
        assert json.dumps(messages)
