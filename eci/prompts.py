"""Documents -> chat messages.

Byte-compatibility with agentere matters here more than anywhere: an identical prompt is what
makes a workere run comparable with the agentere history, and what lets workere reuse the
agentere CoT cache (its keys hash the rendered demo prompt). tests/test_parity.py checks the
bytes against agentere's recorded traces.

Layout of one inference call (single-call prompt):

  System      prompt.system
  Human/AI    one pair per demonstration (a multi-step demo adds one pair per step)
  Human       user_template(doc_text, pair_lines, doc_id) + user_suffix + addenda
  [AI + Tool] reprompt injection, when reprompt.systematic
"""
from __future__ import annotations

import json
import uuid
from typing import List, Optional, Sequence, Tuple

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from eci.config import PromptSpec
from eci.data import Doc, Triple
from eci.labels import is_norel

OPEN_ENDED = "Predict all event mention pairs in the text; omit a pair to mark it norel."

Exchange = Tuple[str, str]           # (human, ai)
Demo = List[Exchange]                # one demonstration: one exchange per prompt step
DemoSet = List[Demo]                 # the demonstrations shown before one call


def pair_lines(doc: Doc, pair_ids: Sequence[Tuple[str, str]], binary_undirected: bool,
               constrain: bool) -> str:
    """The `{pair_lines}` block: an explicit `id ("surface"), id ("surface")` list when the
    dataset defines candidate pairs and the config constrains to them, else the open-ended
    instruction ("omission means norel")."""
    if not (pair_ids and constrain):
        return OPEN_ENDED
    seen, lines = set(), []
    for e1, e2 in pair_ids:
        key = tuple(sorted((e1, e2))) if binary_undirected else (e1, e2)
        if key in seen:
            continue
        seen.add(key)
        s, t = key
        lines.append(f'{s} ("{doc.surfaces.get(s, "")}"), {t} ("{doc.surfaces.get(t, "")}")')
    return "\n".join(lines)


def gold_output(gold: Sequence[Triple], has_pair_list: bool) -> str:
    """Gold as the JSON answer the model is asked for. With a candidate pair list the answer
    labels every listed pair (norel included); without one, omission encodes norel."""
    if has_pair_list:
        items = [{"pair": f"{s},{t}", "label": l} for s, l, t in gold]
    else:
        items = [{"pair": f"{s},{t}", "label": l} for s, l, t in gold if not is_norel(l)]
    return json.dumps(items)


def same_sentence(gold: Sequence[Triple], sentence_of: dict) -> List[Triple]:
    return [(s, l, t) for s, l, t in gold
            if s in sentence_of and t in sentence_of and sentence_of[s] == sentence_of[t]]


def render(prompt: PromptSpec, doc: Doc, pair_ids, *, binary_undirected: bool,
           constrain: bool) -> str:
    """user_template filled for `doc` — the demonstration form, without suffix or addenda."""
    return prompt.user_template.format(
        doc_text=doc.prompt_text,
        pair_lines=pair_lines(doc, pair_ids, binary_undirected, constrain),
        doc_id=doc.id,
    )


def test_prompt(prompt: PromptSpec, doc: Doc, pair_ids, addenda: Sequence[Optional[str]], *,
                binary_undirected: bool, constrain: bool) -> str:
    """The user message of the document under evaluation: the rendered template, the prompt's
    suffix, then each addendum (mention-split block, whitelist, gold count), in that order —
    the narrowest scope restriction is the last instruction the model reads."""
    text = render(prompt, doc, pair_ids, binary_undirected=binary_undirected, constrain=constrain)
    if prompt.user_suffix:
        text = text.rstrip() + "\n\n" + prompt.user_suffix
    for block in addenda:
        if block is not None:
            text = text.rstrip() + "\n\n" + block
    return text


def demo_exchanges(human: str, ai_turns: Sequence[str], steps: Optional[list]) -> Demo:
    """One demonstration: (human, ai) for the first turn, then (step prompt, ai) per step."""
    out = [(human, ai_turns[0])]
    for step, ai in zip(steps or [], ai_turns[1:]):
        out.append((step["prompt"], ai))
    return out


def messages(system: str, demos: Optional[DemoSet], user: str,
             reprompt: str = "") -> List[BaseMessage]:
    msgs: List[BaseMessage] = [SystemMessage(content=system)]
    for demo in demos or []:
        for human, ai in demo:
            msgs.append(HumanMessage(content=human))
            msgs.append(AIMessage(content=ai))
    msgs.append(HumanMessage(content=user))
    if reprompt:
        call_id = uuid.uuid4().hex[:8]
        msgs.append(AIMessage(content="", tool_calls=[{"id": call_id, "name": "reprompt", "args": {}}]))
        msgs.append(ToolMessage(content=reprompt, tool_call_id=call_id))
    return msgs
