"""A scripted chat model for offline tests: no network, deterministic answers."""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Callable, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class Scripted(BaseChatModel):
    """`respond(messages, call_index) -> str | AIMessage | Exception`; exceptions are raised."""

    respond: Any = None
    delay: float = 0.0
    calls: int = 0
    in_flight: int = 0
    peak: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise NotImplementedError("async only")

    async def _agenerate(self, messages: List[BaseMessage], stop=None, run_manager=None, **kwargs):
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            idx = self.calls
            self.calls += 1
            out = self.respond(messages, idx)
        finally:
            self.in_flight -= 1
        if isinstance(out, BaseException):
            raise out
        msg = out if isinstance(out, AIMessage) else AIMessage(
            content=out, response_metadata={"finish_reason": "stop", "token_usage": {"cost": 0.001}},
            usage_metadata={"input_tokens": 100, "output_tokens": 10, "total_tokens": 110})
        return ChatResult(generations=[ChatGeneration(message=msg)])


def gold_answer(docs_by_text, label_filter=("causes", "causedby")) -> Callable:
    """An oracle: answers each evaluated document with its own gold causal pairs (found as the
    last human turn quoting the document — multi-step calls end on a step prompt), and each
    CoT-synthesis request with reasoning ending in the gold it was shown."""
    def respond(messages, idx):
        last = messages[-1].content
        humans = [m.content for m in messages if isinstance(m, HumanMessage)]
        if any(k in last for k in ("The correct label assignments", "Rewrite your response",
                                    "Continue. The correct labels")):
            m = re.search(r"(\[.*\])", "\n".join(humans), re.S)
            return "Step 1 ... reasoning. " + (m.group(1) if m else "[]")
        for h in reversed(humans):
            for text, doc in docs_by_text.items():
                if text in h:
                    pairs = [{"pair": f"{s},{t}", "label": l} for s, l, t in doc.gold
                             if label_filter is None or l in label_filter]
                    return "Reasoning first.\n" + json.dumps(pairs)
        return "[]"
    return respond
